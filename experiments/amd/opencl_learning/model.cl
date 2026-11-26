#define V 4
#define WH (C * H)
#define READOUT (WH + ATTENTION * K * H)
#define BIAS (READOUT + 2 * H)
#define PARAMS (BIAS + 1 + ATTENTION)

kernel void forward_backward(global const float* cameras, global const float* hypotheses,
        global const float* valid, global const float* relation, global const float* labels,
        global const float* theta, global float* gradient, global float* losses,
        global float* prediction, float healthy_weight, float faulty_weight) {
    const size_t frame = get_global_id(0);
    global const float* x = cameras + frame * V * C;
    global const float* k = hypotheses + frame * V * K;
    global const float* live = valid + frame * V;
    global const float* rel = relation + frame * V * V;
    global float* g = gradient + frame * PARAMS;
    float h[V][H], z[V][H], context[V][H], alpha[V][V];
    float dh[V][H] = {{0}}, dz[V][H] = {{0}};
    const float scaling = rsqrt((float)H);
    for (int p = 0; p < PARAMS; ++p) g[p] = 0.0f;
    for (int view = 0; view < V; ++view) {
        for (int q = 0; q < H; ++q) {
            float sum = 0.0f;
            for (int f = 0; f < C; ++f) sum += x[view * C + f] * theta[f * H + q];
            h[view][q] = tanh(sum);
            #if ATTENTION
            sum = 0.0f;
            for (int f = 0; f < K; ++f) sum += k[view * K + f] * theta[WH + f * H + q];
            z[view][q] = tanh(sum);
            #endif
        }
    }
    float total_loss = 0.0f;
    for (int view = 0; view < V; ++view) {
        #if ATTENTION
        float best = -1.0e20f;
        for (int j = 0; j < V; ++j) {
            float dot = 0.0f;
            for (int q = 0; q < H; ++q) dot += h[view][q] * z[j][q];
            alpha[view][j] = live[j] > 0.0f ? dot * scaling + theta[BIAS + 1] * rel[view * V + j] : -1.0e20f;
            best = fmax(best, alpha[view][j]);
        }
        float denom = 0.0f;
        for (int j = 0; j < V; ++j) {
            alpha[view][j] = live[j] > 0.0f ? exp(alpha[view][j] - best) : 0.0f;
            denom += alpha[view][j];
        }
        for (int j = 0; j < V; ++j) alpha[view][j] /= fmax(denom, 1.0e-8f);
        #endif
        float logit = theta[BIAS];
        for (int q = 0; q < H; ++q) {
            context[view][q] = 0.0f;
            for (int j = 0; j < V; ++j) {
                #if ATTENTION
                context[view][q] += alpha[view][j] * z[j][q];
                #else
                context[view][q] += h[j][q] / V;
                #endif
            }
            logit += h[view][q] * theta[READOUT + q] + context[view][q] * theta[READOUT + H + q];
        }
        float probability = 0.5f + 0.5f * tanh(0.5f * logit);
        float target = labels[frame * V + view];
        float weight = target > 0.0f ? healthy_weight : faulty_weight;
        float error = (probability - target) * weight / V;
        prediction[frame * V + view] = probability;
        total_loss += (fmax(logit, 0.0f) - logit * target + log1p(exp(-fabs(logit)))) * weight / V;
        g[BIAS] += error;
        for (int q = 0; q < H; ++q) {
            g[READOUT + q] += error * h[view][q];
            g[READOUT + H + q] += error * context[view][q];
            dh[view][q] += error * theta[READOUT + q];
        }
        for (int j = 0; j < V; ++j) {
            #if ATTENTION
            float score_grad = 0.0f;
            for (int q = 0; q < H; ++q)
                score_grad += (z[j][q] - context[view][q]) * theta[READOUT + H + q];
            score_grad *= error * alpha[view][j];
            g[BIAS + 1] += score_grad * rel[view * V + j];
            for (int q = 0; q < H; ++q) {
                dh[view][q] += score_grad * z[j][q] * scaling;
                dz[j][q] += error * alpha[view][j] * theta[READOUT + H + q] + score_grad * h[view][q] * scaling;
            }
            #else
            for (int q = 0; q < H; ++q) dh[j][q] += error * theta[READOUT + H + q] / V;
            #endif
        }
    }
    losses[frame] = total_loss;
    for (int j = 0; j < V; ++j) {
        for (int q = 0; q < H; ++q) {
            float derivative = dh[j][q] * (1.0f - h[j][q] * h[j][q]);
            for (int f = 0; f < C; ++f) g[f * H + q] += x[j * C + f] * derivative;
            #if ATTENTION
            derivative = dz[j][q] * (1.0f - z[j][q] * z[j][q]);
            for (int f = 0; f < K; ++f) g[WH + f * H + q] += k[j * K + f] * derivative;
            #endif
        }
    }
}

kernel void sgd(global float* theta, global const float* gradient, int frames, float rate) {
    const size_t p = get_global_id(0);
    float total = 0.0f;
    for (int frame = 0; frame < frames; ++frame) total += gradient[frame * PARAMS + p];
    theta[p] -= rate * total / frames;
}
