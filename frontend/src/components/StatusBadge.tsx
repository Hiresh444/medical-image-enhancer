interface Props {
  status: string
}

const STATUS_MAP: Record<string, { cls: string; label: string }> = {
  completed: { cls: 'badge-success', label: 'Completed' },
  running: { cls: 'badge-info', label: 'Running' },
  pending: { cls: 'badge-warning', label: 'Pending' },
  error: { cls: 'badge-danger', label: 'Error' },
}

export default function StatusBadge({ status }: Props) {
  const s = STATUS_MAP[status] ?? { cls: 'badge-secondary', label: status }
  return <span className={`badge ${s.cls}`}>{s.label}</span>
}
