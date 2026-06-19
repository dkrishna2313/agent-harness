import React, { useState } from 'react'
import SimpleList from './fields/SimpleList.jsx'
import DictOfLists from './fields/DictOfLists.jsx'
import ListOfDicts from './fields/ListOfDicts.jsx'
import TextAreaList from './fields/TextAreaList.jsx'

// Fields rendered as raw JSON textarea (advanced/complex)
const ADVANCED_FIELDS = [
  'research_gap_checks',
  'source_quality_hints',
  'web_search',
  'topic_categories',
]

// Dict of lists where values are regex strings (use TextAreaList per section)
const METRIC_PATTERN_FIELDS = ['metric_patterns']

// Topic section checks: dict[str, list[str]] with 4 fixed cols
const TOPIC_SECTION_CHECKS_FIELDS = ['topic_section_checks']

function Section({ title, children }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-5 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="font-semibold text-slate-700">{title}</span>
        <span className="text-slate-400 text-sm">{open ? '▼' : '▶'}</span>
      </button>
      {open && <div className="px-5 py-4 space-y-5">{children}</div>}
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide">{label}</label>
      {children}
    </div>
  )
}

function TextInput({ value, onChange, placeholder }) {
  return (
    <input
      type="text"
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full border border-slate-200 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
    />
  )
}

function JsonTextarea({ value, onChange }) {
  const [err, setErr] = useState(null)
  const str = (() => {
    try { return JSON.stringify(value, null, 2) } catch { return '' }
  })()
  const [text, setText] = useState(str)

  const handleChange = (e) => {
    setText(e.target.value)
    try {
      const parsed = JSON.parse(e.target.value)
      onChange(parsed)
      setErr(null)
    } catch (ex) {
      setErr(ex.message)
    }
  }

  return (
    <div className="space-y-1">
      <textarea
        value={text}
        onChange={handleChange}
        rows={8}
        className={`w-full text-xs font-mono border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200 resize-y ${
          err ? 'border-red-300' : 'border-slate-200'
        }`}
        spellCheck={false}
      />
      {err && <p className="text-xs text-red-500">{err}</p>}
    </div>
  )
}

// DictOfLists but each value section uses TextAreaList instead of SimpleList
function DictOfTextAreaLists({ value = {}, onChange }) {
  const [collapsed, setCollapsed] = useState({})
  const [newKey, setNewKey] = useState('')
  const [showAdd, setShowAdd] = useState(false)

  const dict = value && typeof value === 'object' && !Array.isArray(value) ? value : {}

  const updateKey = (key, list) => onChange({ ...dict, [key]: list })
  const deleteKey = (key) => { const n = { ...dict }; delete n[key]; onChange(n) }
  const addKey = () => {
    const k = newKey.trim()
    if (!k || k in dict) return
    onChange({ ...dict, [k]: [] }); setNewKey(''); setShowAdd(false)
  }
  const toggle = (key) => setCollapsed((c) => ({ ...c, [key]: !c[key] }))

  return (
    <div className="space-y-2">
      {Object.entries(dict).map(([key, list]) => (
        <div key={key} className="border border-slate-200 rounded-lg overflow-hidden">
          <div
            className="flex items-center justify-between px-3 py-2 bg-slate-50 cursor-pointer select-none"
            onClick={() => toggle(key)}
          >
            <span className="text-sm font-medium text-slate-700">{key}</span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">{(list || []).length} patterns</span>
              <button onClick={(e) => { e.stopPropagation(); deleteKey(key) }} className="text-slate-300 hover:text-red-500 text-base leading-none">×</button>
              <span className="text-slate-400 text-xs">{collapsed[key] ? '▶' : '▼'}</span>
            </div>
          </div>
          {!collapsed[key] && (
            <div className="px-3 py-3">
              <TextAreaList value={list || []} onChange={(v) => updateKey(key, v)} />
            </div>
          )}
        </div>
      ))}
      {showAdd ? (
        <div className="flex gap-2">
          <input type="text" value={newKey} onChange={(e) => setNewKey(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') addKey() }} placeholder="New metric key..." className="flex-1 text-sm border border-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-300" autoFocus />
          <button onClick={addKey} className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700">Add</button>
          <button onClick={() => setShowAdd(false)} className="text-sm px-3 py-1.5 border border-slate-200 rounded hover:bg-slate-50">Cancel</button>
        </div>
      ) : (
        <button onClick={() => setShowAdd(true)} className="text-sm text-blue-600 hover:underline">+ Add metric</button>
      )}
    </div>
  )
}

// topic_section_checks: dict[str, list[str]] but each value has 4 fixed string slots
function TopicSectionChecks({ value = {}, onChange }) {
  const dict = value && typeof value === 'object' && !Array.isArray(value) ? value : {}
  const [newKey, setNewKey] = useState('')
  const [showAdd, setShowAdd] = useState(false)

  const COLS = ['section_key', 'Section Title', 'missing_section_code', 'missing_citations_code']

  const updateCell = (topic, idx, val) => {
    const row = [...(dict[topic] || ['', '', '', ''])]
    row[idx] = val
    onChange({ ...dict, [topic]: row })
  }
  const deleteKey = (key) => { const n = { ...dict }; delete n[key]; onChange(n) }
  const addKey = () => {
    const k = newKey.trim()
    if (!k || k in dict) return
    onChange({ ...dict, [k]: ['', '', '', ''] }); setNewKey(''); setShowAdd(false)
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">Topic</th>
              {COLS.map((c) => (
                <th key={c} className="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">{c}</th>
              ))}
              <th className="w-10" />
            </tr>
          </thead>
          <tbody>
            {Object.entries(dict).length === 0 && (
              <tr><td colSpan={6} className="text-center text-slate-400 py-4 text-sm">No entries yet.</td></tr>
            )}
            {Object.entries(dict).map(([topic, row]) => (
              <tr key={topic} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/50">
                <td className="px-3 py-1.5 font-medium text-slate-700 text-sm whitespace-nowrap">{topic}</td>
                {COLS.map((_, idx) => (
                  <td key={idx} className="px-3 py-1.5">
                    <input
                      type="text"
                      value={(row || [])[idx] || ''}
                      onChange={(e) => updateCell(topic, idx, e.target.value)}
                      className="w-full text-sm border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    />
                  </td>
                ))}
                <td className="px-2 py-1.5 text-center">
                  <button onClick={() => deleteKey(topic)} className="text-slate-300 hover:text-red-500 text-lg leading-none">×</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {showAdd ? (
        <div className="flex gap-2">
          <input type="text" value={newKey} onChange={(e) => setNewKey(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') addKey() }} placeholder="Topic name..." className="flex-1 text-sm border border-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-300" autoFocus />
          <button onClick={addKey} className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700">Add</button>
          <button onClick={() => setShowAdd(false)} className="text-sm px-3 py-1.5 border border-slate-200 rounded hover:bg-slate-50">Cancel</button>
        </div>
      ) : (
        <button onClick={() => setShowAdd(true)} className="text-sm text-blue-600 hover:underline">+ Add topic</button>
      )}
    </div>
  )
}

export default function ProfileEditor({ profileName, data, onChange }) {
  const set = (key) => (val) => onChange({ ...data, [key]: val })

  const entitySchema = [
    { key: 'name', label: 'Name', type: 'text' },
    { key: 'signals', label: 'Signals (comma-separated)', type: 'chips' },
  ]

  const sqSchema = [
    { key: 'signals', label: 'Signals', type: 'chips' },
    { key: 'type', label: 'Type', type: 'text' },
    { key: 'score', label: 'Score (1-5)', type: 'number' },
    { key: 'label', label: 'Label', type: 'text' },
  ]

  // Collect any fields not explicitly handled so we don't lose them
  const HANDLED = new Set([
    'name', 'description',
    'domain_signals', 'coverage_topics', 'memo_section_hints',
    'perspectives', 'evaluator_topic_terms', 'coverage_gap_keywords',
    'topic_query_expansions', 'required_topic_terms', 'metric_anchor_queries',
    'topic_keywords', 'contradiction_topics',
    'entity_patterns', 'source_quality_rules',
    'metric_patterns', 'topic_section_checks',
    ...ADVANCED_FIELDS,
  ])

  const unknownFields = Object.keys(data || {}).filter((k) => !HANDLED.has(k))

  return (
    <div className="space-y-4">
      {/* Basic Info */}
      <Section title="Basic Info">
        <Field label="Name">
          <TextInput value={data.name} onChange={set('name')} placeholder="Profile name" />
        </Field>
        <Field label="Description">
          <TextInput value={data.description} onChange={set('description')} placeholder="Short description" />
        </Field>
      </Section>

      {/* Topic Keywords */}
      <Section title="Topic Keywords">
        <Field label="topic_keywords">
          <DictOfLists value={data.topic_keywords} onChange={set('topic_keywords')} />
        </Field>
        <Field label="domain_signals">
          <SimpleList value={data.domain_signals} onChange={set('domain_signals')} />
        </Field>
        <Field label="coverage_topics">
          <SimpleList value={data.coverage_topics} onChange={set('coverage_topics')} />
        </Field>
      </Section>

      {/* Coverage & Gaps */}
      <Section title="Coverage & Gaps">
        <Field label="coverage_gap_keywords">
          <DictOfLists value={data.coverage_gap_keywords} onChange={set('coverage_gap_keywords')} />
        </Field>
        <Field label="required_topic_terms">
          <DictOfLists value={data.required_topic_terms} onChange={set('required_topic_terms')} />
        </Field>
        <Field label="memo_section_hints">
          <SimpleList value={data.memo_section_hints} onChange={set('memo_section_hints')} />
        </Field>
        <Field label="topic_section_checks">
          <TopicSectionChecks value={data.topic_section_checks} onChange={set('topic_section_checks')} />
        </Field>
      </Section>

      {/* Perspectives */}
      <Section title="Perspectives">
        <Field label="perspectives">
          <DictOfLists value={data.perspectives} onChange={set('perspectives')} />
        </Field>
        <Field label="contradiction_topics">
          <DictOfLists value={data.contradiction_topics} onChange={set('contradiction_topics')} />
        </Field>
      </Section>

      {/* Entities & Metrics */}
      <Section title="Entities & Metrics">
        <Field label="entity_patterns">
          <ListOfDicts
            value={data.entity_patterns}
            onChange={set('entity_patterns')}
            schema={entitySchema}
            addLabel="Add entity pattern"
          />
        </Field>
        <Field label="metric_patterns">
          <DictOfTextAreaLists value={data.metric_patterns} onChange={set('metric_patterns')} />
        </Field>
        <Field label="metric_anchor_queries">
          <DictOfLists value={data.metric_anchor_queries} onChange={set('metric_anchor_queries')} />
        </Field>
      </Section>

      {/* Source Quality */}
      <Section title="Source Quality">
        <Field label="source_quality_rules">
          <ListOfDicts
            value={data.source_quality_rules}
            onChange={set('source_quality_rules')}
            schema={sqSchema}
            addLabel="Add rule"
          />
        </Field>
      </Section>

      {/* Evaluator Settings */}
      <Section title="Evaluator Settings">
        <Field label="evaluator_topic_terms">
          <DictOfLists value={data.evaluator_topic_terms} onChange={set('evaluator_topic_terms')} />
        </Field>
        <Field label="topic_query_expansions">
          <DictOfLists value={data.topic_query_expansions} onChange={set('topic_query_expansions')} />
        </Field>
      </Section>

      {/* Advanced */}
      <Section title="Advanced (JSON)">
        {ADVANCED_FIELDS.map((key) => (
          data[key] !== undefined ? (
            <Field key={key} label={key}>
              <JsonTextarea value={data[key]} onChange={set(key)} />
            </Field>
          ) : null
        ))}
      </Section>

      {/* Unknown fields — don't lose them */}
      {unknownFields.length > 0 && (
        <Section title="Other Fields">
          {unknownFields.map((key) => (
            <Field key={key} label={key}>
              <JsonTextarea value={data[key]} onChange={set(key)} />
            </Field>
          ))}
        </Section>
      )}
    </div>
  )
}
