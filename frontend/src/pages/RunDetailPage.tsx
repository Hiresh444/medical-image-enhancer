import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getRun, getReport, getBeforeAfterUrl, type RunDetail } from '../api/client'
import StatusBadge from '../components/StatusBadge'
import MetricsTable from '../components/MetricsTable'
import ChatPanel from '../components/ChatPanel'
import MarkdownViewer from '../components/MarkdownViewer'
import JsonViewer from '../components/JsonViewer'
import LogsAccordion from '../components/LogsAccordion'

const TABS = [
  'Overview',
  'Metrics',
  'Plan JSON',
  'Validation',
  'Visuals',
  'Report',
  'Logs',
  'Chat',
] as const
type Tab = (typeof TABS)[number]

export default function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [data, setData] = useState<RunDetail | null>(null)
  const [report, setReport] = useState<string>('')
  const [tab, setTab] = useState<Tab>('Overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!runId) return
    setLoading(true)
    Promise.all([getRun(runId), getReport(runId).catch(() => '')])
      .then(([run, md]) => {
        setData(run)
        setReport(md)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [runId])

  if (loading) {
    return (
      <div className="text-center mt-2">
        <span className="spinner" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card" style={{ borderColor: 'var(--danger)' }}>
        <p style={{ color: 'var(--danger)' }}>
          {error || 'Run not found'}
        </p>
        <Link to="/runs" className="btn btn-secondary mt-1">Back to Runs</Link>
      </div>
    )
  }

  const val = data.validation ?? {}

  return (
    <div>
      {/* Header */}
      <div className="flex gap-2" style={{ alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <h1 style={{ margin: 0 }}>Run {data.run_id}</h1>
        <StatusBadge status={data.status} />
        <span className="text-muted" style={{ fontSize: '0.85rem' }}>
          {data.timestamp ? new Date(data.timestamp).toLocaleString() : ''}
        </span>
        <span className="text-muted" style={{ fontSize: '0.85rem' }}>
          | {data.input_filename}
        </span>
        {data.genai_model && (
          <span className="badge badge-info">{data.genai_model}</span>
        )}
      </div>

      {/* Tabs */}
      <div className="tabs">
        {TABS.map((t) => (
          <div
            key={t}
            className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </div>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'Overview' && <OverviewTab data={data} />}
      {tab === 'Metrics' && <MetricsTab data={data} />}
      {tab === 'Plan JSON' && <JsonViewer data={data.plan_json} title="Enhancement Plan" />}
      {tab === 'Validation' && <ValidationTab val={val} />}
      {tab === 'Visuals' && <VisualsTab runId={data.run_id} />}
      {tab === 'Report' && <MarkdownViewer markdown={report} />}
      {tab === 'Logs' && <LogsAccordion logs={data.agent_logs ?? []} />}
      {tab === 'Chat' && (
        <ChatPanel runId={data.run_id} initialHistory={data.chat_history ?? []} />
      )}
    </div>
  )
}

/* ===== Sub-tab components ===== */

function OverviewTab({ data }: { data: RunDetail }) {
  const expl = data.explainability ?? {}
  return (
    <div>
      {/* Issues */}
      <div className="card">
        <h3>Detected Issues</h3>
        {(data.issues ?? []).length === 0 ? (
          <p className="text-muted">No issues detected.</p>
        ) : (
          <div className="flex gap-1" style={{ flexWrap: 'wrap' }}>
            {data.issues.map((issue, i) => (
              <span key={i} className="badge badge-warning">{issue}</span>
            ))}
          </div>
        )}
      </div>

      {/* Applied Ops */}
      <div className="card">
        <h3>Applied Operations</h3>
        {(data.applied_ops ?? []).length === 0 ? (
          <p className="text-muted">No operations applied.</p>
        ) : (
          <ol>
            {data.applied_ops.map((op, i) => (
              <li key={i} style={{ marginBottom: '0.3rem' }}>{op}</li>
            ))}
          </ol>
        )}
      </div>

      {/* Metadata */}
      {data.metadata_summary && Object.keys(data.metadata_summary).length > 0 && (
        <div className="card">
          <h3>Metadata (non-PHI)</h3>
          <table>
            <tbody>
              {Object.entries(data.metadata_summary).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ fontWeight: 600, width: '200px' }}>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Explainability */}
      {expl && typeof expl === 'object' && Object.keys(expl).length > 0 && (
        <div className="card">
          <h3>Explainability</h3>
          {Object.entries(expl).map(([k, v]) => {
            if (!v) return null
            return (
              <div key={k} style={{ marginBottom: '0.75rem' }}>
                <strong style={{ textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</strong>
                {Array.isArray(v) ? (
                  <ul>
                    {(v as string[]).map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ marginTop: '0.2rem' }}>{String(v)}</p>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function MetricsTab({ data }: { data: RunDetail }) {
  const before = data.metrics_before ?? {}
  const after = data.metrics_after ?? {}
  return (
    <div className="card">
      <h3>Before / After Metrics</h3>
      <MetricsTable before={before} after={after} />
    </div>
  )
}

function ValidationTab({ val }: { val: Record<string, unknown> }) {
  if (!val || Object.keys(val).length === 0) {
    return <p className="text-muted">No validation data.</p>
  }

  const passes = val.passes as boolean | undefined
  const ssim = val.ssim as number | undefined
  const psnr = val.psnr as number | undefined
  const qi = val.quality_improvement as number | undefined

  const gainKeys = [
    'contrast_gain',
    'sharpness_gain',
    'noise_change',
    'entropy_change',
    'snr_change',
    'cnr_change',
    'edge_density_change',
    'histogram_spread_change',
    'local_contrast_change',
    'gradient_strength_change',
    'gradient_entropy_change',
  ]

  return (
    <div>
      {/* Pass/Fail */}
      <div className="card">
        <h3>Validation Result</h3>
        <div className="flex gap-3" style={{ flexWrap: 'wrap', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>
              {passes === true ? '✅ PASS' : passes === false ? '❌ FAIL' : '—'}
            </span>
          </div>
          <div>
            <label>SSIM</label>
            <span style={{ fontFamily: 'var(--font-mono)', display: 'block' }}>
              {ssim?.toFixed(4) ?? '—'}
              {val.meets_ssim !== undefined && (
                <span className={`badge ${val.meets_ssim ? 'badge-success' : 'badge-danger'}`} style={{ marginLeft: '0.5rem' }}>
                  {val.meets_ssim ? '≥0.70' : '<0.70'}
                </span>
              )}
            </span>
          </div>
          <div>
            <label>PSNR</label>
            <span style={{ fontFamily: 'var(--font-mono)', display: 'block' }}>
              {psnr?.toFixed(2) ?? '—'} dB
              {val.meets_psnr !== undefined && (
                <span className={`badge ${val.meets_psnr ? 'badge-success' : 'badge-danger'}`} style={{ marginLeft: '0.5rem' }}>
                  {val.meets_psnr ? '≥22' : '<22'}
                </span>
              )}
            </span>
          </div>
          <div>
            <label>Quality Improvement</label>
            <span style={{ fontFamily: 'var(--font-mono)', display: 'block' }}>
              {qi !== undefined ? `${(qi * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* NIQE */}
      <div className="card">
        <h3>NIQE Approximation</h3>
        <table>
          <thead>
            <tr><th>Metric</th><th>Before</th><th>After</th><th>Improved?</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>NIQE (lower = better)</td>
              <td style={{ fontFamily: 'var(--font-mono)' }}>{(val.niqe_before as number)?.toFixed(4) ?? '—'}</td>
              <td style={{ fontFamily: 'var(--font-mono)' }}>{(val.niqe_after as number)?.toFixed(4) ?? '—'}</td>
              <td>{val.niqe_improved ? '✅ Yes' : '❌ No'}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Gains */}
      <div className="card">
        <h3>Component Gains / Changes</h3>
        <table>
          <thead>
            <tr><th>Component</th><th>Value</th></tr>
          </thead>
          <tbody>
            {gainKeys.map((k) => {
              const v = val[k] as number | undefined
              if (v === undefined) return null
              return (
                <tr key={k}>
                  <td>{k.replace(/_/g, ' ')}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {typeof v === 'number' ? v.toFixed(6) : String(v)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function VisualsTab({ runId }: { runId: string }) {
  const url = getBeforeAfterUrl(runId)
  return (
    <div className="card">
      <h3>Before / After</h3>
      <img
        src={url}
        alt="Before and After comparison"
        style={{ maxWidth: '100%', borderRadius: '8px', border: '1px solid var(--border)' }}
        onError={(e) => {
          ;(e.target as HTMLImageElement).style.display = 'none'
          const p = document.createElement('p')
          p.className = 'text-muted'
          p.textContent = 'Before/after image not available.'
          ;(e.target as HTMLImageElement).parentElement?.appendChild(p)
        }}
      />
    </div>
  )
}
