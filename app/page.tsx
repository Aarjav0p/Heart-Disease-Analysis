'use client';

import { useMemo, useState } from 'react';
import {
  Activity, BarChart3, BrainCircuit, ChevronDown, CircleHelp, Gauge,
  HeartPulse, LayoutDashboard, Menu, RotateCcw, Settings2, ShieldCheck,
  Sparkles, TrendingUp, UserRound, X
} from 'lucide-react'

type Patient = {
  age: number
  sex: string
  chestPain: string
  bp: number
  cholesterol: number
  sugar: string
  ecg: string
  maxRate: number
  angina: string
  oldpeak: number
  slope: string
}

type MetricSet = {
  model: string
  accuracy: number
  balancedAccuracy: number
  precision: number
  recall: number
  f1: number
  rocAuc: number
}

type Prediction = {
  prediction: number
  label: 'Positive' | 'Negative'
  probability: number
}

type Results = {
  consensusProbability: number
  logisticRegression: Prediction
  randomForest: Prediction
  metrics: MetricSet[]
  disclaimer: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? ''

const defaults: Patient = {
  age: 54,
  sex: 'Male',
  chestPain: 'Typical angina',
  bp: 132,
  cholesterol: 246,
  sugar: 'Normal',
  ecg: 'Normal',
  maxRate: 150,
  angina: 'No',
  oldpeak: 1.2,
  slope: 'Flat'
}

const fields: { key: keyof Patient; label: string; hint: string }[] = [
  { key: 'age', label: 'Age', hint: 'years' },
  { key: 'bp', label: 'Resting blood pressure', hint: 'mmHg' },
  { key: 'cholesterol', label: 'Cholesterol', hint: 'mg/dL' },
  { key: 'maxRate', label: 'Maximum heart rate', hint: 'bpm' },
  { key: 'oldpeak', label: 'Oldpeak', hint: 'ST depression' },
]

export default function Page() {
  const [patient, setPatient] = useState(defaults)
  const [results, setResults] = useState<Results | null>(null)
  const [loading, setLoading] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = (key: keyof Patient, value: string) => {
    setPatient((p) => ({
      ...p,
      [key]: ['age', 'bp', 'cholesterol', 'maxRate', 'oldpeak'].includes(key) ? Number(value) : value
    }))
  }

  const analyze = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patient),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'Prediction failed.')
      setResults(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to reach the Python model service.')
    } finally {
      setLoading(false)
    }
  }

  const risk = results?.consensusProbability ?? 0
  const riskLabel = useMemo(
    () => risk > 50 ? 'Elevated model probability' : 'Lower model probability',
    [risk]
  )

  const metric = (name: string) => {
    const match = results?.metrics?.find((m) => m.model === name)
    return match
  }

  const lr = metric('Logistic Regression')
  const rf = metric('Random Forest')

  return (
    <main className="min-h-screen overflow-x-hidden bg-background text-foreground">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
      <aside className={`sidebar ${navOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <span className="brand-mark"><HeartPulse /></span>
          <span>Cardio<span>Lens</span></span>
          <button className="icon-btn mobile-close" onClick={() => setNavOpen(false)} aria-label="Close navigation"><X /></button>
        </div>
        <div className="workspace-label">Workspace</div>
        <nav className="nav-list">
          <a className="nav-item active" href="#overview"><LayoutDashboard /> Overview</a>
          <a className="nav-item" href="#patient"><UserRound /> Patient intake</a>
          <a className="nav-item" href="#analytics"><BarChart3 /> Model analytics</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="mini-status"><span className="status-dot" /> Python model service connected</div>
          <a className="nav-item" href="#settings"><Settings2 /> Settings</a>
          <div className="profile">
            <div className="avatar">DR</div>
            <div><strong>Dr. Rivera</strong><small>Cardiology team</small></div>
            <ChevronDown />
          </div>
        </div>
      </aside>

      <div className="shell">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setNavOpen(true)} aria-label="Open navigation"><Menu /></button>
          <div>
            <p className="eyebrow">CLINICAL INTELLIGENCE <span>•</span> 24 AUG 2026</p>
            <h1>Heart disease assessment</h1>
          </div>
          <div className="top-actions">
            <div className="secure-pill"><ShieldCheck /> Local model workspace</div>
            <button className="icon-btn" aria-label="Help"><CircleHelp /></button>
          </div>
        </header>

        <section className="hero" id="overview">
          <div>
            <span className="section-kicker"><Sparkles /> MODEL SNAPSHOT</span>
            <h2>Evaluate patient risk<br /><em>with clarity.</em></h2>
            <p>Enter clinical features to compare predictions from Logistic Regression and Random Forest.</p>
          </div>
          <div className="hero-orbit">
            <div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" />
            <div className="orbit-core"><Activity /><strong>11</strong><small>features tracked</small></div>
          </div>
        </section>

        <div className="dashboard-grid" id="patient">
          <section className="glass-card intake-card">
            <div className="card-heading">
              <div>
                <span className="section-kicker">01 / PATIENT PROFILE</span>
                <h3>Clinical features</h3>
                <p>Use the most recent resting measurements.</p>
              </div>
              <span className="completion">11 / 11 ready</span>
            </div>

            <div className="field-grid">
              {fields.map((f) => (
                <label className="field" key={f.key}>
                  <span>{f.label}</span>
                  <div className="input-wrap">
                    <input type="number" value={patient[f.key] as number} onChange={(e) => update(f.key, e.target.value)} />
                    <small>{f.hint}</small>
                  </div>
                </label>
              ))}
              <SelectField label="Sex" value={patient.sex} options={['Male', 'Female']} onChange={(v) => update('sex', v)} />
              <SelectField label="Chest pain type" value={patient.chestPain} options={['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic']} onChange={(v) => update('chestPain', v)} />
              <SelectField label="Fasting blood sugar" value={patient.sugar} options={['Normal', 'Elevated']} onChange={(v) => update('sugar', v)} />
              <SelectField label="Resting ECG" value={patient.ecg} options={['Normal', 'ST-T abnormality', 'LV hypertrophy']} onChange={(v) => update('ecg', v)} />
              <SelectField label="Exercise angina" value={patient.angina} options={['No', 'Yes']} onChange={(v) => update('angina', v)} />
              <SelectField label="ST slope" value={patient.slope} options={['Upsloping', 'Flat', 'Downsloping']} onChange={(v) => update('slope', v)} />
            </div>

            <div className="card-footer">
              <span className="privacy-note"><ShieldCheck /> Data stays in your local session</span>
              <div className="footer-actions">
                <button className="text-btn" onClick={() => { setPatient(defaults); setResults(null); setError(null) }}>
                  <RotateCcw /> Reset
                </button>
                <button className="primary-btn" onClick={analyze} disabled={loading}>
                  <BrainCircuit /> {loading ? 'Evaluating...' : 'Analyze patient'}
                </button>
              </div>
            </div>
            {error && <p className="error-banner">{error}</p>}
          </section>

          <section className="results-column" id="analytics">
            <div className="glass-card risk-card">
              <div className="card-heading">
                <div><span className="section-kicker">02 / MODEL OUTPUT</span><h3>Prediction result</h3></div>
                <span className="live-badge"><span /> LIVE</span>
              </div>

              <div className="risk-main">
                <div className="score-ring" style={{ '--score': `${risk}%` } as React.CSSProperties}>
                  <div><strong>{results ? `${risk.toFixed(1)}%` : '—'}</strong><small>model consensus</small></div>
                </div>
                <div className="risk-copy">
                  <span className="risk-chip"><TrendingUp /> {results ? riskLabel : 'Awaiting patient analysis'}</span>
                  <p>Simple mean of the two model probabilities. This is model output, not a clinical diagnosis.</p>
                </div>
              </div>

              <div className="prediction-row">
                <Prediction label="Logistic Regression" prediction={results?.logisticRegression} />
                <Prediction label="Random Forest" prediction={results?.randomForest} />
              </div>
            </div>

            <div className="glass-card metrics-card">
              <div className="card-heading">
                <div><span className="section-kicker">03 / VALIDATION</span><h3>Model performance</h3></div>
                <span className="muted-label">{results ? 'Held-out test set' : 'Run an analysis to load metrics'}</span>
              </div>

              <div className="metric-grid">
                {[
                  ['Accuracy', (lr?.accuracy ?? 0) * 100],
                  ['Precision', (lr?.precision ?? 0) * 100],
                  ['Recall', (lr?.recall ?? 0) * 100],
                  ['F1 score', (lr?.f1 ?? 0) * 100],
                  ['Balanced acc.', (lr?.balancedAccuracy ?? 0) * 100],
                  ['ROC-AUC', (lr?.rocAuc ?? 0) * 100],
                ].map(([label, value]) => (
                  <div className="metric" key={label as string}>
                    <small>Logistic Regression · {label}</small>
                    <strong>{results ? `${Number(value).toFixed(1)}%` : '—'}</strong>
                    <div className="metric-bar"><i style={{ width: `${Math.min(100, Number(value))}%` }} /></div>
                  </div>
                ))}
              </div>

              <div className="model-summary-row">
                <div><small>Logistic Regression AUC</small><strong>{lr ? `${(lr.rocAuc * 100).toFixed(1)}%` : '—'}</strong></div>
                <div><small>Random Forest AUC</small><strong>{rf ? `${(rf.rocAuc * 100).toFixed(1)}%` : '—'}</strong></div>
              </div>

              {results?.disclaimer && <p className="disclaimer">{results.disclaimer}</p>}
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="select-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o}>{o}</option>)}
        </select>
        <ChevronDown />
      </div>
    </label>
  )
}

function Prediction({ label, prediction }: { label: string; prediction?: Prediction }) {
  const isPositive = prediction?.prediction === 1
  return (
    <div className="prediction">
      <div className="model-icon"><Gauge /></div>
      <div>
        <small>{label}</small>
        <strong>{prediction ? prediction.label : '—'}</strong>
      </div>
      <span className={isPositive ? 'prediction-alert' : 'prediction-safe'}>
        {prediction ? `${prediction.probability.toFixed(1)}%` : '—'}
      </span>
    </div>
  )
}
