import React, { useState } from 'react'

export default function ProfileList({ profiles, selected, onSelect, onCreate, onDelete, loading }) {
  const [showModal, setShowModal] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(null)

  const handleCreate = () => {
    if (!newName.trim()) return
    onCreate(newName.trim(), newDesc.trim())
    setNewName('')
    setNewDesc('')
    setShowModal(false)
  }

  const handleDelete = (name) => {
    onDelete(name)
    setConfirmDelete(null)
  }

  return (
    <>
      <div className="w-64 min-w-[14rem] bg-slate-100 border-r border-slate-200 flex flex-col h-full">
        <div className="px-4 py-3 border-b border-slate-200">
          <h1 className="text-sm font-semibold text-slate-700 uppercase tracking-widest">Profiles</h1>
        </div>

        <div className="flex-1 overflow-y-auto py-2">
          {loading ? (
            <div className="px-4 py-3 text-sm text-slate-400">Loading...</div>
          ) : profiles.length === 0 ? (
            <div className="px-4 py-3 text-sm text-slate-400">No profiles found.</div>
          ) : (
            profiles.map((name) => (
              <div
                key={name}
                className={`group flex items-center justify-between px-4 py-2 cursor-pointer rounded mx-2 my-0.5 transition-colors ${
                  selected === name
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-700 hover:bg-slate-200'
                }`}
                onClick={() => onSelect(name)}
              >
                <span className="text-sm truncate">{name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); setConfirmDelete(name) }}
                  className={`opacity-0 group-hover:opacity-100 text-base leading-none transition-opacity ${
                    selected === name ? 'text-blue-200 hover:text-white' : 'text-slate-300 hover:text-red-500'
                  }`}
                  aria-label="Delete profile"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>

        <div className="px-4 py-3 border-t border-slate-200">
          <button
            onClick={() => setShowModal(true)}
            className="w-full text-sm py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors font-medium"
          >
            + New Profile
          </button>
        </div>
      </div>

      {/* New profile modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 w-96 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">New Profile</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Name (used as filename)</label>
                <input
                  autoFocus
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
                  placeholder="e.g. wind_energy"
                  className="w-full border border-slate-200 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Description</label>
                <input
                  type="text"
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
                  placeholder="Short description..."
                  className="w-full border border-slate-200 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
              </div>
            </div>
            <div className="flex gap-3 pt-1">
              <button
                onClick={handleCreate}
                disabled={!newName.trim()}
                className="flex-1 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 text-sm font-medium"
              >
                Create
              </button>
              <button
                onClick={() => setShowModal(false)}
                className="flex-1 py-2 border border-slate-200 text-slate-600 rounded hover:bg-slate-50 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirm modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 w-80 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">Delete Profile</h2>
            <p className="text-sm text-slate-600">
              Are you sure you want to delete <strong>{confirmDelete}</strong>? This cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleDelete(confirmDelete)}
                className="flex-1 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm font-medium"
              >
                Delete
              </button>
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 py-2 border border-slate-200 text-slate-600 rounded hover:bg-slate-50 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
