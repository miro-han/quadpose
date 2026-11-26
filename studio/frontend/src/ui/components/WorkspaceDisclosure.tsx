import { ChevronDown } from "lucide-react";
import type {
  ButtonHTMLAttributes,
  ComponentPropsWithoutRef,
  ReactNode,
} from "react";

type DisclosureProps = Omit<ComponentPropsWithoutRef<"details">, "title"> & {
  title: string;
  description?: string;
  icon?: ReactNode;
};

export function WorkspaceDisclosure({
  title,
  description,
  icon,
  className = "",
  children,
  ...props
}: DisclosureProps) {
  return (
    <details {...props} className={`workspace-disclosure ${className}`}>
      <summary>
        {icon && (
          <span aria-hidden="true" className="disclosure-icon">
            {icon}
          </span>
        )}
        <span className="disclosure-heading">
          <span>{title}</span>
          {description && <small>{description}</small>}
        </span>
        <ChevronDown
          className="disclosure-chevron"
          size={16}
          aria-hidden="true"
        />
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

type WorkspaceButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "quiet";
};

export function WorkspaceButton({
  variant = "secondary",
  className = "",
  type = "button",
  ...props
}: WorkspaceButtonProps) {
  return (
    <button
      {...props}
      type={type}
      data-variant={variant}
      className={`workspace-button ${className}`}
    />
  );
}
