import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getRuns, type RunSummary } from '../api/client'
import StatusBadge from '../components/StatusBadge'

export default function RunsListPage() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getRuns()
      .then(setRuns)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="text-center mt-2">
        <span className="spinner" />
      </div>
    )
  }

  if (error) {
    return <div className="card" style={{ borderColor: 'var(--danger)' }}>Error: {error}</div>
  }

  return (
    <div>
      <h1 style={{ marginBottom: '1.5rem' }}>Pipeline Runs</h1>
      {runs.length === 0 ? (
        <div className="card text-center">
          <p className="text-muted">No runs yet. Upload a DICOM file to get started.</p>
          <Link to="/" className="btn btn-primary mt-1">Upload</Link>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table>
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Timestamp</th>
                <th>File</th>
                <th>Status</th>
                <th>Issues</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} style={{ cursor: 'pointer' }}>
                  <td>
                    <Link to={`/runs/${r.run_id}`} style={{ fontFamily: 'var(--font-mono)' }}>
                      {r.run_id}
                    </Link>
                  </td>
                  <td style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    {r.timestamp ? new Date(r.timestamp).toLocaleString() : '—'}
                  </td>
                  <td>{r.input_filename || '—'}</td>
                  <td><StatusBadge status={r.status} /></td>
                  <td>
                    {(r.issues ?? []).length > 0 ? (
                      (r.issues ?? []).map((issue, i) => (
                        <span key={i} className="badge badge-warning" style={{ marginRight: '0.25rem' }}>
                          {issue}
                        </span>
                      ))
                    ) : (
                      <span className="text-muted">none</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                    {r.genai_model || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
