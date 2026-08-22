// @vitest-environment jsdom
/**
 * Version-family status aggregation (dsh-webchatlike): the sidebar hides
 * regenerate/edit fork sessions, so the visible original row must fold the
 * family's live status (running / completed / pending interaction) onto
 * itself — otherwise a conversation running through a hidden fork shows no
 * status indicator at all. These specs drive the derivation with a recorded
 * `dsh-webchatlike:version-tree` ledger and assert the folded row state.
 */
import { afterEach, describe, expect, it } from 'vitest'
import type {
  SessionId, SessionListState, SessionSummary, WorkspaceId, WorkspaceView,
} from '@deepseek-ai/dsh-client-runtime/client'
import {
  deriveArchived, deriveFlat, deriveGroups, recentVersionForkOf,
  UNGROUPED_KEY,
} from '../src/client/tree.ts'

const sid = (id: string) => id as SessionId
const wid = (id: string) => id as WorkspaceId
const summary = (id: string, updatedAt: number, cwd?: string): SessionSummary => ({
  id: sid(id), displayTitle: id, running: false, blank: false,
  updatedAt, ...(cwd === undefined ? {} : { cwd }),
})
const list = (...items: SessionSummary[]): SessionListState => ({
  ids: items.map(item => item.id),
  byId: Object.fromEntries(items.map(item => [item.id, item])),
  current: undefined,
  phase: 'ready', subagentsByParent: {}, jobsBySession: {}, currentAddress: undefined,
})
const workspace = (id: string, sessionIds: string[], title = id): WorkspaceView => ({
  workspaceId: wid(id), path: `/projects/${id}`, title,
  sessionIds: sessionIds.map(sid), createdAt: '2026-01-01T00:00:00.000Z', updatedAt: '2026-01-01T00:00:00.000Z',
})
const view = (expandedGroups: readonly string[] = []) => ({ expandedGroups })
const noArchive: readonly SessionId[] = []
const archived = (...ids: string[]): readonly SessionId[] => ids.map(sid)

describe('version-family status aggregation (dsh-webchatlike)', () => {
  const VERSION_TREE_KEY = 'dsh-webchatlike:version-tree'
  const record = (atSeq: string, original: string, versions: string[]) => {
    const raw = localStorage.getItem(VERSION_TREE_KEY)
    const tree = raw === null ? {} : JSON.parse(raw) as Record<string, { original: string; versions: string[] }>
    tree[atSeq] = { original, versions }
    localStorage.setItem(VERSION_TREE_KEY, JSON.stringify(tree))
  }
  afterEach(() => { localStorage.clear() })

  it('folds a hidden running fork into its original row and hides the fork row', () => {
    record('10', 'orig', ['fork1'])
    const orig = summary('orig', 1)
    const fork1 = { ...summary('fork1', 2), parentId: orig.id, running: true }
    const sessions = list(orig, fork1)
    const groups = deriveGroups(sessions, [workspace('first', ['orig', 'fork1'])], noArchive, view(['first']))
    expect(groups[0]!.sessions.map(session => session.id)).toEqual([orig.id])
    expect(groups[0]!.sessions[0]).toMatchObject({ running: true, completed: false })
    expect(deriveFlat(sessions, noArchive)[0]).toMatchObject({ running: true })
  })

  it('folds completion reminder and pending interaction from hidden forks', () => {
    record('10', 'orig', ['fork1'])
    const orig = summary('orig', 1)
    const fork1 = {
      ...summary('fork1', 2), parentId: orig.id, completed: true, pendingInteraction: 'approval' as const,
    }
    const sessions = list(orig, fork1)
    const node = deriveFlat(sessions, noArchive)[0]!
    expect(node.completed).toBe(true)
    expect(node.pendingInteraction).toBe('approval')
  })

  it('folds forks-of-forks into the root original row', () => {
    record('10', 'orig', ['fork1'])
    record('20', 'fork1', ['fork2'])
    const orig = summary('orig', 1)
    const fork1 = { ...summary('fork1', 2), parentId: orig.id }
    const fork2 = { ...summary('fork2', 3), parentId: fork1.id, running: true }
    const sessions = list(orig, fork1, fork2)
    const node = deriveFlat(sessions, noArchive).find(row => row.id === orig.id)!
    expect(node.running).toBe(true)
  })

  it('folds hidden fork status into archived rows', () => {
    record('10', 'orig', ['fork1'])
    const orig = summary('orig', 1)
    const fork1 = { ...summary('fork1', 2), parentId: orig.id, running: true }
    const sessions = list(orig, fork1)
    const rows = deriveArchived(sessions, archived('orig'))
    expect(rows[0]).toMatchObject({ id: orig.id, running: true })
  })

  it('keeps own status when no fork is recorded', () => {
    const orig = { ...summary('orig', 1), running: true }
    const sessions = list(orig)
    expect(deriveFlat(sessions, noArchive)[0]!.running).toBe(true)
  })

  it('keeps the ungrouped bucket working with a recorded family', () => {
    record('10', 'orig', ['fork1'])
    const orig = summary('orig', 1)
    const fork1 = { ...summary('fork1', 2), parentId: orig.id, running: true }
    const sessions = list(orig, fork1)
    const groups = deriveGroups(sessions, [], noArchive, view([UNGROUPED_KEY]))
    expect(groups[0]!.sessions.map(session => session.id)).toEqual([orig.id])
    expect(groups[0]!.sessions[0]!.running).toBe(true)
  })

  it('recentVersionForkOf returns the newest hidden fork newer than the row', () => {
    record('10', 'orig', ['fork1'])
    record('20', 'orig', ['fork2'])
    const orig = summary('orig', 1)
    const fork1 = { ...summary('fork1', 2), parentId: orig.id }
    const fork2 = { ...summary('fork2', 3), parentId: orig.id }
    const sessions = list(orig, fork1, fork2)
    expect(recentVersionForkOf(orig.id, sessions.byId)).toBe(fork2.id)
  })

  it('recentVersionForkOf is undefined when no fork is newer than the row', () => {
    record('10', 'orig', ['fork1'])
    const orig = { ...summary('orig', 5), running: true }
    const fork1 = { ...summary('fork1', 2), parentId: orig.id }
    const sessions = list(orig, fork1)
    expect(recentVersionForkOf(orig.id, sessions.byId)).toBeUndefined()
  })

  it('recentVersionForkOf is undefined without a recorded family', () => {
    const orig = summary('orig', 1)
    const sessions = list(orig)
    expect(recentVersionForkOf(orig.id, sessions.byId)).toBeUndefined()
  })
})
