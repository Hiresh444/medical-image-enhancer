interface Props {
  before: Record<string, number>
  after: Record<string, number>
}

const METRIC_LABELS: Record<string, string> = {
  sigma: 'Noise (sigma)',
  lap_var: 'Laplacian Variance',
  std: 'Contrast (std)',
  pct_low: 'Clip Low %',
  pct_high: 'Clip High %',
  entropy: 'Entropy',
  edge_density: 'Edge Density',
  gradient_mag_mean: 'Gradient Mean',
  gradient_mag_std: 'Gradient Std',
  snr_proxy: 'SNR Proxy',
  cnr_proxy: 'CNR Proxy',
  laplacian_energy: 'Laplacian Energy',
  histogram_spread: 'Histogram Spread',
  local_contrast_std: 'Local Contrast Std',
  gradient_strength: 'Gradient Strength',
  gradient_entropy: 'Gradient Entropy',
}

// Metrics where lower is better
const LOWER_IS_BETTER = new Set(['sigma', 'pct_low', 'pct_high'])

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(4)
  return v.toFixed(6)
}

export default function MetricsTable({ before, after }: Props) {
  const keys = Object.keys(before).filter((k) => k in METRIC_LABELS)

  return (
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Before</th>
          <th>After</th>
          <th>Delta</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => {
          const b = before[k] ?? 0
          const a = after[k] ?? 0
          const delta = a - b
          const lowerBetter = LOWER_IS_BETTER.has(k)
          const improved = lowerBetter ? delta < 0 : delta > 0
          const cls =
            Math.abs(delta) < 1e-8
              ? 'delta-neutral'
              : improved
              ? 'delta-up'
              : 'delta-down'
          const arrow = Math.abs(delta) < 1e-8 ? '—' : improved ? '▲' : '▼'

          return (
            <tr key={k}>
              <td>{METRIC_LABELS[k] ?? k}</td>
              <td style={{ fontFamily: 'var(--font-mono)' }}>{fmt(b)}</td>
              <td style={{ fontFamily: 'var(--font-mono)' }}>{fmt(a)}</td>
              <td className={cls} style={{ fontFamily: 'var(--font-mono)' }}>
                {arrow} {fmt(delta)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
