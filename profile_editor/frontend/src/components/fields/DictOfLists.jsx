import React, { useState } from 'react'
import SimpleList from './SimpleList.jsx'

// dict[str, list[str]] — collapsible sections with tag chips per key
export default function DictOfLists({ value = {}, onChange }) {
  const [collapsed, setCollapsed] = useState({})
  const [newKey, setNewKey] = useState('')
  const [showAdd, setShowAdd] = useState(false)

  const dict = value && typeof value === 'object' && !Array.isArray(value) ? value : {}

  const updateKey = (key, list) => onChange({ ...dict, [key]: list })

  const deleteKey = (key) => {
    const next = { ...dict }
    delete next[key]
    onChange(next)
  }

  const addKey = () => {
    const k = newKey.trim()
    if (!k || k in dict) return
    onChange({ ...dict, [k]: [] })
    setNewKey('')
    setShowAdd(false)
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
              <span className="text-xs text-slate-400">{(list || []).length} items</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteKey(key) }}
                className="text-slate-300 hover:text-red-500 text-base leading-none"
                aria-label="Delete section"
              >
                ×
              </button>
              <span className="text-slate-400 text-xs">{collapsed[key] ? '▶' : '▼'}</span>
            </div>
          </div>
          {!collapsed[key] && (
            <div className="px-3 py-3">
              <SimpleList
                value={list || []}
                onChange={(v) => updateKey(key, v)}
                placeholder={`Add to ${key}...`}
              />
            </div>
          )}
        </div>
      ))}

      {showAdd ? (
        <div className="flex gap-2">
          <input
            type="text"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addKey() }}
            placeholder="New section key..."
            className="flex-1 text-sm border border-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-300"
            autoFocus
          />
          <button onClick={addKey} className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700">Add</button>
          <button onClick={() => setShowAdd(false)} className="text-sm px-3 py-1.5 border border-slate-200 rounded hover:bg-slate-50">Cancel</button>
        </div>
      ) : (
        <button
          onClick={() => setShowAdd(true)}
          className="text-sm text-blue-600 hover:underline"
        >
          + Add section
        </button>
      )}
    </div>
  )
}
