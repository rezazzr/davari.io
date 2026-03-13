export const variantStyles = {
  info: "bg-info/10 text-info hover:bg-info/20",
  warning: "bg-warning/10 text-warning hover:bg-warning/20",
  primary: "bg-primary/10 text-primary hover:bg-primary/20",
};

interface CollapsibleButtonProps {
  buttonLabel: string;
  variant?: "info" | "warning" | "primary";
  isOpen: boolean;
  onClick: () => void;
}

export default function CollapsibleButton({
  buttonLabel,
  variant = "info",
  isOpen,
  onClick,
}: CollapsibleButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`mr-2 mt-2 inline-flex items-center gap-1 rounded-md px-3 py-1.5 text-sm font-medium transition-all active:scale-95 ${variantStyles[variant]} ${isOpen ? "ring-2 ring-current/30" : ""}`}
    >
      {buttonLabel}
    </button>
  );
}
