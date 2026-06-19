import React, { useState, useEffect, useCallback, useRef } from 'react'
import ProfileList from './components/ProfileList.jsx'
import ProfileEditor from './components/ProfileEditor.jsx'

const API = '/api'

function Toast({ message, type, onDismiss }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 3000)
    return () => clearTimeout(t)
  }, [onDismiss])

  const colors =
    type === 'success'
      ? 'bg-green-50 border-green-200 text-green-800'
      : 'bg-red-50 border-red-200 text-red-800'

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 px-4 py-3 rounded-lg border shadow-lg text-sm font-medium flex items-center gap-2 ${colors}`}
    >
      {type === 'success' ? '✓' : '✕'} {message}
      <button onClick={onDismiss} className="ml-2 opacity-50 hover:opacity-100 text-base leading-none">×</button>
    </div>
  )
}

export default function App() {
  const [profiles, setProfiles] = useState([])
  const [selected, setSelected] = useState(null)
  const [profileData, setProfileData] = useState(null)
  const [savedData, setSavedData] = useState(null)
  const [loadingList, setLoadingList] = useState(true)
  const [loadingProfile, setLoadingProfile] = useState(false)
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState(null)

  const isDirty = profileData !== null && JSON.stringify(profileData) !== JSON.stringify(savedData)

  const showToast = (message, type = 'success') => setToast({ message, type })

  // Load profile list
  const loadProfiles = useCallback(async () => {
    setLoadingList(true)
    try {
      const res = await fetch(`${API}/profiles`)
      const data = await res.json()
      setProfiles(data)
    } catch (e) {
      showToast('Failed to load profiles', 'error')
    } finally {
      setLoadingList(false)
    }
  }, [])

  useEffect(() => { loadProfiles() }, [loadProfiles])

  // Load a profile
  const selectProfile = useCallback(async (name) => {
    setSelected(name)
    setLoadingProfile(true)
    setProfileData(null)
    setSavedData(null)
    try {
      const res = await fetch(`${API}/profiles/${encodeURIComponent(name)}`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setProfileData(data)
      setSavedData(JSON.parse(JSON.stringify(data)))
    } catch (e) {
      showToast(`Failed to load profile: ${e.message}`, 'error')
    } finally {
      setLoadingProfile(false)
    }
  }, [])

  // Save profile
  const saveProfile = async () => {
    if (!selected || !profileData) return
    setSaving(true)
    try {
      const res = await fetch(`${API}/profiles/${encodeURIComponent(selected)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileData),
      })
      if (!res.ok) throw new Error(await res.text())
      setSavedData(JSON.parse(JSON.stringify(profileData)))
      showToast('Saved successfully')
    } catch (e) {
      showToast(`Save failed: ${e.message}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  // Create profile
  const createProfile = async (name, description) => {
    try {
      const res = await fetch(`${API}/profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      })
      if (!res.ok) throw new Error(await res.text())
      await loadProfiles()
      await selectProfile(name)
      showToast(`Profile "${name}" created`)
    } catch (e) {
      showToast(`Create failed: ${e.message}`, 'error')
    }
  }

  // Delete profile
  const deleteProfile = async (name) => {
    try {
      const res = await fetch(`${API}/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(await res.text())
      if (selected === name) {
        setSelected(null)
        setProfileData(null)
        setSavedData(null)
      }
      await loadProfiles()
      showToast(`Profile "${name}" deleted`)
    } catch (e) {
      showToast(`Delete failed: ${e.message}`, 'error')
    }
  }

  // Keyboard shortcut: Cmd/Ctrl + S to save
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        if (isDirty) saveProfile()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isDirty, profileData, selected])

  return (
    <div className="flex h-screen overflow-hidden bg-white font-sans">
      <ProfileList
        profiles={profiles}
        selected={selected}
        onSelect={selectProfile}
        onCreate={createProfile}
        onDelete={deleteProfile}
        loading={loadingList}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white shrink-0">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-slate-800">
              {selected ? selected : 'Select a profile'}
              {isDirty && <span className="ml-2 text-yellow-500 text-sm font-normal">● Unsaved</span>}
            </h2>
          </div>
          {selected && (
            <button
              onClick={saveProfile}
              disabled={saving || !isDirty}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                isDirty
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {!selected && (
            <div className="h-full flex items-center justify-center text-slate-400 text-sm">
              Select a profile from the sidebar to begin editing.
            </div>
          )}
          {selected && loadingProfile && (
            <div className="h-full flex items-center justify-center text-slate-400 text-sm">
              Loading…
            </div>
          )}
          {selected && !loadingProfile && profileData && (
            <ProfileEditor
              profileName={selected}
              data={profileData}
              onChange={setProfileData}
            />
          )}
        </div>
      </div>

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onDismiss={() => setToast(null)}
        />
      )}
    </div>
  )
}
