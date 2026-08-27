// ChatView: the default conversation view — one stable keyed parent list over
// final business Nodes, plus paging, pending steering and bottom-follow.
// Each row dispatches through 'conversation.chat.node'; ui-tool owns the
// tool-call renderer and its recursive root/subcall composition. A Host
// open-path refusal from the injected opener is an in-page dialog here.
//
// Scroll: when nested under `[data-conversation-scroll]` (active conversation
// column), that host is the scrollport and this view is flow content; when
// mounted alone (unit tests), `.scroll` owns overflow. Bottom-follow and
// prepend anchoring always target the resolved scrollport.
//
// Render economics: order changes only when rows enter, leave or move. Each
// ChatNodeSeat subscribes to one Node key, so Assistant deltas and Tool
// lifecycle updates replace only their own row without remounting it.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { ConversationTimelineSnapshot } from '@deepseek-ai/dsh-client-runtime/client'
import { Button, IconChevronDownOutline14, Modal } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ChatViewSlotProps, RenderMessageImages } from '../contract/slots.ts'
import { PendingSteeringBubble } from './MessageItem.tsx'
import { ChatNodeSeat } from './ChatNodeSeat.tsx'
import { formatRunDuration } from './message-chrome.ts'
import css from './ChatView.module.css'

const FOLLOW_THRESHOLD = 24

/** Active column host when present; otherwise the view-local scroller. */
function scrollerOf(from: HTMLElement): HTMLElement {
  return (from.closest('[data-conversation-scroll]')) ?? from
}

interface PagingAnchor {
  /** Stable node/call identity, independent of boundary-spanning group keys. */
  key: string
  /** Row top relative to the scrollport after the latest user scroll. */
  top: number
}

/** Find an already-rendered settled row without interpolating a selector. */
function anchorElement(list: HTMLElement, key: string): HTMLElement | null {
  for (const row of list.querySelectorAll<HTMLElement>('[data-chat-anchor-key]')) {
    if (row.dataset.chatAnchorKey === key) return row
  }
  return null
}

/** Row position in scrollport coordinates (viewport-independent). */
function flowTop(row: HTMLElement, scrollport: HTMLElement): number {
  return row.getBoundingClientRect().top - scrollport.getBoundingClientRect().top
}

/**
 * Per-row layout metrics for windowed rendering. Measured once after the
 * window's first full render settles, then maintained incrementally: appended
 * rows are estimated at the tail, prepends rebuild the whole index. Offsets
 * are stored in scrollport-content coordinates (independent of the current
 * scrollTop), so paging resolves rows by binary search over plain numbers
 * instead of probing DOM geometry on every scroll event.
 */
interface RowMetrics {
  /** node/call key → offset from the scrollport content top, in pixels. */
  offsets: Map<string, number>
  /** node/call key → last measured height, in pixels (estimated until measured). */
  heights: Map<string, number>
}

/** Height estimate for rows that were never measured (off-window prepends). */
const ROW_HEIGHT_ESTIMATE = 160

/** Extra rows kept mounted above/below the viewport during windowed rendering. */
const WINDOW_BUFFER_ROWS = 12

/** Windowed rendering needs a real layout engine: jsdom (unit tests)
 * implements neither observer API, so it stays on the full-row path and the
 * suite keeps asserting the complete list. */
const CAN_WINDOW = typeof IntersectionObserver !== 'undefined' && typeof ResizeObserver !== 'undefined'

/** Select a visible stable node/call identity, falling back only when layout
 * has not exposed a visible box yet. When row metrics are available (large
 * windows in windowed-rendering mode) this binary-searches measured offsets,
 * which costs no DOM reads; otherwise it scans mounted rows with an early
 * exit. `elementsFromPoint` is deliberately not used: it performs a full
 * document hit-test on every call, which blocked the main thread for ~200ms
 * per scroll event on large windows. */
function pagingAnchor(
  list: HTMLElement,
  scrollport: HTMLElement,
  metrics: RowMetrics | null,
): PagingAnchor | HTMLElement | null {
  if (metrics !== null && metrics.offsets.size > 0) {
    const keys = [...metrics.offsets.keys()]
    const target = scrollport.scrollTop
    // First row whose measured bottom exceeds the viewport top — the topmost
    // row at/inside the viewport. Pure index math: no DOM geometry reads, so
    // the hot path never forces layout or materializes skipped rows.
    let lo = 0
    let hi = keys.length - 1
    let idx = keys.length
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const key = keys[mid] as string
      const bottom = (metrics.offsets.get(key) as number) + (metrics.heights.get(key) ?? ROW_HEIGHT_ESTIMATE)
      if (bottom <= target) {
        lo = mid + 1
      } else {
        idx = mid
        hi = mid - 1
      }
    }
    if (idx >= keys.length) idx = keys.length - 1
    if (idx < 0) return null
    const key = keys[idx] as string
    // Position-only anchor: no DOM probe, so paging never materializes an
    // off-screen row. Streaming-height drift self-corrects on the next
    // measured pass (prepend rebuilds; appends extend from the tail).
    return { key, top: (metrics.offsets.get(key) as number) - scrollport.scrollTop }
  }
  const viewport = scrollport.getBoundingClientRect()
  const composer = scrollport.querySelector<HTMLElement>('[data-composer-seat]')
  const visibleBottom = composer?.getBoundingClientRect().top ?? viewport.bottom
  const rows = list.querySelectorAll<HTMLElement>('[data-chat-anchor-key]')
  // Rows are mounted in flow order, so the first one crossing the viewport
  // top is the reader anchor; early-exit keeps this proportional to the rows
  // above the viewport instead of a full pass.
  for (const row of rows) {
    const rect = row.getBoundingClientRect()
    if (rect.bottom > viewport.top && rect.top < visibleBottom) return row
  }
  return rows[0] ?? null
}

type ChatScrollPosition = NonNullable<ReturnType<ChatViewSlotProps['chatScroll']['read']>>

/** Capture a reflow-resistant reader position from the current rendered window. */
function scrollPosition(
  list: HTMLElement,
  scrollport: HTMLElement,
  metrics: RowMetrics | null,
): ChatScrollPosition | null {
  const anchor = pagingAnchor(list, scrollport, metrics)
  if (anchor === null) return null
  if (anchor instanceof HTMLElement) {
    const anchorKey = anchor.dataset.chatAnchorKey
    if (anchorKey === undefined) return null
    return {
      anchorKey,
      anchorTop: flowTop(anchor, scrollport),
      scrollTop: scrollport.scrollTop,
    }
  }
  return {
    anchorKey: anchor.key,
    anchorTop: anchor.top,
    scrollTop: scrollport.scrollTop,
  }
}

/** Rebuild the sorted offset index from the current height table. O(n) pure
 * math — no DOM reads — so it is cheap enough to run after streaming height
 * changes and appends. */
function rebuildOffsets(metrics: RowMetrics, keys: readonly string[]): void {
  let acc = 0
  for (const key of keys) {
    metrics.offsets.set(key, acc)
    acc += metrics.heights.get(key) ?? ROW_HEIGHT_ESTIMATE
  }
}

/**
 * Compute the mounted row slice for a scrollTop plus the exact pixel heights
 * of the two inert spacers standing in for the skipped rows. Binary search
 * over the measured offsets keeps this O(log n) with no DOM geometry reads.
 */
function computeWindow(
  scrollTop: number,
  clientHeight: number,
  keys: readonly string[],
  metrics: RowMetrics,
): { start: number; end: number; topPad: number; bottomPad: number } {
  const firstPast = (target: number): number => {
    let lo = 0
    let hi = keys.length - 1
    let idx = keys.length
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const key = keys[mid] as string
      const bottom = (metrics.offsets.get(key) as number) + (metrics.heights.get(key) ?? ROW_HEIGHT_ESTIMATE)
      if (bottom <= target) {
        lo = mid + 1
      } else {
        idx = mid
        hi = mid - 1
      }
    }
    return idx
  }
  const start = Math.max(0, firstPast(scrollTop) - WINDOW_BUFFER_ROWS)
  const end = Math.min(keys.length, firstPast(scrollTop + clientHeight) + WINDOW_BUFFER_ROWS)
  const topPad = start === 0
    ? 0
    : (metrics.offsets.get(keys[start] as string) as number)
  const lastKey = keys[keys.length - 1]
  const total = lastKey === undefined
    ? 0
    : (metrics.offsets.get(lastKey) as number) + (metrics.heights.get(lastKey) ?? ROW_HEIGHT_ESTIMATE)
  const bottomPad = end >= keys.length
    ? 0
    : total - (metrics.offsets.get(keys[end] as string) as number)
  return { start, end, topPad, bottomPad }
}

/** Host/OS refusal text for the file-open dialog; empty throws keep a locale fallback. */
function openFailureMessage(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : String(error)
  return message === '' ? fallback : message
}

/** ProducedFiles opens the session workspace as `.`. */
function isFolderOpenPath(path: string): boolean {
  return path === '.'
}

function runningTurnStartTime(timeline: ConversationTimelineSnapshot): number | null {
  let latest: number | null = null
  for (const turn of timeline.turns.values()) {
    if (turn.status === 'open' && turn.start !== undefined) latest = turn.start.time
  }
  return latest
}

/** Turn-level model activity label retained across first-token, tool, and streaming phases. */
function TurnStatus({ startTime, t }: {
  /** The running turn's logged `turn/start` time; null falls back to mount
   *  time when that boundary is outside the window. */
  startTime: number | null
  /** The owning view's locale seat. */
  t: ChatViewSlotProps['t']
}) {
  const [mountedAt] = useState(() => Date.now())
  // Anchored to turn/start so a mid-turn reload keeps the real
  // elapsed time and the final footer's Ran-for label matches this clock.
  const anchor = startTime ?? mountedAt
  const [elapsedMs, setElapsedMs] = useState(() => Math.max(0, Date.now() - anchor))
  useEffect(() => {
    const tick = (): void => {
      setElapsedMs(Math.max(0, Date.now() - anchor))
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => { clearInterval(id) }
  }, [anchor])
  // Short turns keep the plain label; the clock only appears once the turn
  // has clearly been running for a while.
  const showClock = elapsedMs >= 15_000
  return (
    <div className={css.turnStatus} role="status" aria-live="polite">
      Deep diving...
      {showClock && (
        <span className={css.turnStatusClock} aria-hidden>
          {formatRunDuration(elapsedMs, t)}
        </span>
      )}
    </div>
  )
}

/**
 * The chat view slot entry: pure component over the composed props; each
 * ordered business Node crosses the keyed renderer seat.
 */
export function ChatView({
  useSession, useSessions, useStore, renderSlot, sessionId, openFile, loadOlder, loadImage, inspectCall, chatScroll, forkAt,
  fileMentions, t,
}: ChatViewSlotProps) {
  const order = useSession(s => s.chat.order)
  const nodeStore = useSession(s => s.chat.nodes)
  const timeline = useSession(s => s.chat.timeline)
  const inbox = useSession(s => s.queue)
  // Workspace root off the session list row: path summaries display relative to it.
  const cwd = useSessions(s => s.byId[sessionId]?.cwd)
  const running = useSession(s => s.running)
  const openState = useSession(s => s.openState)
  const openError = useSession(s => s.openError)
  const hasMore = useSession(s => s.hasMore)
  const loadingOlder = useSession(s => s.loadingOlder)
  const selectedCallId = useStore(s => s.selection?.callId)
  const [fileOpenError, setFileOpenError] = useState<{ path: string; message: string } | null>(null)
  const [fileOpenBusy, setFileOpenBusy] = useState(false)
  // Close/retry must ignore a settlement that started before the latest
  // gesture; otherwise a cancelled in-flight refusal reopens the dialog.
  const fileOpenRequest = useRef(0)

  const requestOpenFile = useCallback((path: string) => {
    const id = ++fileOpenRequest.current
    setFileOpenBusy(true)
    void openFile(path).then(
      () => {
        if (id !== fileOpenRequest.current) return
        setFileOpenError(null)
        setFileOpenBusy(false)
      },
      (error: unknown) => {
        if (id !== fileOpenRequest.current) return
        setFileOpenError({
          path,
          message: openFailureMessage(
            error,
            t(isFolderOpenPath(path) ? 'fileOpen.folderUnknown' : 'fileOpen.unknown'),
          ),
        })
        setFileOpenBusy(false)
      },
    )
  }, [openFile, t])

  const closeFileOpenError = useCallback(() => {
    fileOpenRequest.current += 1
    setFileOpenError(null)
    setFileOpenBusy(false)
  }, [])

  const pendingSteering = useMemo(
    () => inbox.filter(item => item.placement === 'steering'),
    [inbox],
  )
  const renderMessageImages = useCallback<RenderMessageImages>(
    owner => renderSlot('conversation.message.images', { ...owner, loadImage }),
    [loadImage, renderSlot],
  )
  const runningTurnStart = useMemo(() => runningTurnStartTime(timeline), [timeline])

  const listRef = useRef<HTMLDivElement | null>(null)
  const columnRef = useRef<HTMLDivElement | null>(null)
  const atBottomRef = useRef(true)
  const [atBottom, setAtBottom] = useState(true)
  /** Last position delivered or written on the main thread. */
  const observedTopRef = useRef(0)
  /** Paging anchor: semantic row/position at click, updated by reader scrolls
   * while the request is pending and restored after the prepend lands. */
  const anchorRef = useRef<PagingAnchor | null>(null)
  const firstSeqRef = useRef<number | null>(null)
  const openedRef = useRef(false)
  const lastKeyRef = useRef<string | null>(null)
  const lastSteeringIdRef = useRef<string | null>(null)
  /** Flow tip signature — follow-scroll only when this moves, never on a
   *  scroll-driven at-bottom chrome re-render (which would snap inertial
   *  scrolls the rest of the way to the floor). */
  const followSigRef = useRef<string | null>(null)

  const firstKey = order[0]
  const firstSeq = firstKey === undefined ? null : nodeStore.get(firstKey)?.anchorSeq ?? null
  const lastKey = order.at(-1) ?? null
  const lastNode = lastKey === null ? undefined : nodeStore.get(lastKey)
  const lastSteeringId = pendingSteering[pendingSteering.length - 1]?.id ?? null
  const followSig = `${openState}:${firstSeq}:${lastKey}:${order.length}:${running ? 1 : 0}:${lastSteeringId ?? ''}`

  /** Windowed rendering (virtualization): the slice of `order` currently
   * mounted plus the exact pixel heights of the two inert spacers standing in
   * for the skipped rows. null = not virtualized (jsdom has no layout, and
   * short windows gain nothing). */
  const [windowed, setWindowed] = useState<{ start: number; end: number; topPad: number; bottomPad: number } | null>(null)
  const metricsRef = useRef<RowMetrics | null>(null)
  /** Whether the window has been measured at least once (real heights). */
  const measuredRef = useRef(false)

  // Render-phase estimate seeding: from the first frame that has enough rows
  // (real browsers only), seed an estimate-only offset index and render only
  // the tail slice, so opening/switching a heavy session never mounts the
  // whole window just to measure it. The measurement pass replaces the
  // estimates right after. Writing the ref here is idempotent (same order →
  // same result), so double-invocation in StrictMode is harmless.
  if (metricsRef.current === null && CAN_WINDOW && openState === 'open' && order.length > WINDOW_BUFFER_ROWS * 3) {
    const heights = new Map<string, number>()
    for (const key of order) heights.set(key, ROW_HEIGHT_ESTIMATE)
    const metrics: RowMetrics = { offsets: new Map(), heights }
    rebuildOffsets(metrics, order)
    metricsRef.current = metrics
  }
  /** The window to render: the measured one once available, otherwise the
   * estimate-mode tail slice (open pins to the bottom). Derived in render so
   * the first frame with a large order never renders the full list. */
  const renderWindow = windowed !== null ? windowed : (
    metricsRef.current !== null && CAN_WINDOW
      ? (() => {
        const n = order.length
        const start = Math.max(0, n - (WINDOW_BUFFER_ROWS * 2 + 24))
        return { start, end: n, topPad: start * ROW_HEIGHT_ESTIMATE, bottomPad: 0 }
      })()
      : null
  )

  const toBottom = (el: HTMLElement): void => {
    anchorRef.current = null
    el.scrollTop = el.scrollHeight
    observedTopRef.current = el.scrollTop
    atBottomRef.current = true
    setAtBottom(true)
    chatScroll.save(null)
  }

  /** Compute the mounted window from the reader's intent: while pinned to the
   * bottom the window always targets the current floor (scrollHeight is the
   * live DOM floor, independent of the actual scrollTop), so streaming and
   * history growth cannot leave the mounted slice and the scroll position
   * disagreeing. */
  const windowAt = useCallback((
    metrics: RowMetrics,
    el: HTMLElement,
    intentBottom: boolean,
  ): { start: number; end: number; topPad: number; bottomPad: number } => {
    const top = intentBottom ? Math.max(0, el.scrollHeight - el.clientHeight) : el.scrollTop
    return computeWindow(top, el.clientHeight, order, metrics)
  }, [order])

  /** (Re)build the offset index from the current height table: every row
   * starts at the estimate, mounted rows (only the window is mounted) get
   * their measured height, then offsets are rebuilt as an order-prefix sum.
   * Pure math plus one pass over the ~24 mounted rows — cheap enough to run
   * after prepends and as the windowed baseline.
   */
  const rebuildMetrics = useCallback((local: HTMLElement): void => {
    const column = local.querySelector<HTMLElement>('[data-chat-flow]')
    if (column === null || order.length === 0) return
    const heights = new Map<string, number>()
    for (const key of order) heights.set(key, ROW_HEIGHT_ESTIMATE)
    for (const child of column.children) {
      if (!(child instanceof HTMLElement)) continue
      const key = child.dataset.chatAnchorKey
      if (key === undefined) continue
      heights.set(key, child.getBoundingClientRect().height)
    }
    const metrics: RowMetrics = { offsets: new Map(), heights }
    rebuildOffsets(metrics, order)
    metricsRef.current = metrics
  }, [order])

  // Windowed-rendering measurement: after the estimate-mode window settles,
  // measure its mounted rows into the offset index and re-window against the
  // measured floor. Never enabled in jsdom (CAN_WINDOW) or short windows, so
  // unit-test behavior is unchanged.
  useEffect(() => {
    if (openState !== 'open' || measuredRef.current) return
    const local = listRef.current
    if (local === null) return
    const el = scrollerOf(local)
    if (el.scrollHeight <= el.clientHeight + 1) return // no scrollable extent
    let cancelled = false
    const raf = requestAnimationFrame(() => {
      if (cancelled) return
      requestAnimationFrame(() => {
        if (cancelled) return
        const node = listRef.current
        if (node === null) return
        const port = scrollerOf(node)
        measuredRef.current = true
        rebuildMetrics(node)
        // The open flow may still be pinning to the floor while the window
        // loads; re-pin against the post-measurement floor so the mounted
        // slice and scrollTop agree from the first windowed frame.
        if (atBottomRef.current) toBottom(port)
        const metrics = metricsRef.current
        if (metrics !== null) setWindowed(windowAt(metrics, port, atBottomRef.current))
      })
    })
    return () => { cancelled = true; cancelAnimationFrame(raf) }
  }, [openState, firstSeq, lastKey, order.length])

  useLayoutEffect(() => {
    const local = listRef.current
    /* v8 ignore next -- ref-null guard: React attaches the ref before layout effects run. */
    if (local === null) return
    const el = scrollerOf(local)
    // Open completed: jump to the bottom once — unless a scroll position
    // survives from a previous mount (view-tab switch away and back), which
    // is restored instead of snapping the reader back to the floor.
    if (openState === 'open' && !openedRef.current) {
      openedRef.current = true
      const saved = chatScroll.read()
      if (saved === null) {
        toBottom(el)
      } else {
        el.scrollTop = saved.scrollTop
        const row = anchorElement(local, saved.anchorKey)
        if (row !== null) el.scrollTop += flowTop(row, el) - saved.anchorTop
        observedTopRef.current = el.scrollTop
        const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= FOLLOW_THRESHOLD + 1
        atBottomRef.current = isAtBottom
        setAtBottom(isAtBottom)
        const normalized = isAtBottom ? null : scrollPosition(local, el, metricsRef.current)
        if (isAtBottom) chatScroll.save(null)
        else if (normalized !== null) chatScroll.save(normalized)
      }
      firstSeqRef.current = firstSeq
      lastKeyRef.current = lastKey
      lastSteeringIdRef.current = lastSteeringId
      followSigRef.current = followSig
      return
    }
    // Prepend (head seq decreased): preserve the same settled row at the
    // position established by the reader's latest scroll. This excludes
    // unrelated tail/composer growth while the request was in flight.
    if (anchorRef.current !== null && firstSeq !== null && firstSeqRef.current !== null && firstSeq < firstSeqRef.current) {
      const anchor = anchorRef.current
      anchorRef.current = null
      const row = anchorElement(local, anchor.key)
      if (row !== null) el.scrollTop += flowTop(row, el) - anchor.top
      observedTopRef.current = el.scrollTop
      // Prepend inserts off-viewport top rows: rebuild the offset index and
      // the mounted window so paging and spacers see the new geometry.
      if (metricsRef.current !== null) {
        rebuildMetrics(local)
        setWindowed(windowAt(metricsRef.current, el, false))
      }
      firstSeqRef.current = firstSeq
      /* v8 ignore next -- ?? arm: a prepend adds nodes, so the flow list here is never empty. */
      lastKeyRef.current = lastKey
      lastSteeringIdRef.current = lastSteeringId
      followSigRef.current = followSig
      return
    }
    firstSeqRef.current = firstSeq
    // Own words must be visible: a new trailing user node force-scrolls
    // (send lives in the composer, so arrival is detected here, not armed there).
    const appendedUser = lastKey !== lastKeyRef.current && lastNode?.kind === 'user'
    const appendedSteering = lastSteeringId !== null && lastSteeringId !== lastSteeringIdRef.current
    const tipMoved = followSigRef.current !== followSig
    // New trailing rows while windowed: extend the offset index from the
    // previous measured tail row (heights estimated; the next measurement
    // pass corrects them once the row renders), so paging and the bottom
    // spacer stay right without a DOM pass per message.
    if (metricsRef.current !== null && lastKey !== null && lastKey !== lastKeyRef.current) {
      const metrics = metricsRef.current
      let cursorKey: string | null = null
      for (const key of order) {
        if (metrics.offsets.has(key)) {
          cursorKey = key
          continue
        }
        if (cursorKey !== null) {
          const prevHeight = metrics.heights.get(cursorKey) ?? ROW_HEIGHT_ESTIMATE
          metrics.offsets.set(key, (metrics.offsets.get(cursorKey) as number) + prevHeight)
          metrics.heights.set(key, ROW_HEIGHT_ESTIMATE)
        }
        cursorKey = key
      }
      setWindowed(windowAt(metrics, el, atBottomRef.current))
    }
    // While windowed, keep the height table in sync with streaming growth of
    // mounted rows: measure the rendered slice (cheap — only the window is
    // mounted) and rebuild offsets when anything moved.
    if (metricsRef.current !== null) {
      const metrics = metricsRef.current
      const column = local.querySelector<HTMLElement>('[data-chat-flow]')
      if (column !== null) {
        let changed = false
        for (const child of column.children) {
          if (!(child instanceof HTMLElement)) continue
          const key = child.dataset.chatAnchorKey
          if (key === undefined) continue
          const h = child.getBoundingClientRect().height
          if (Math.abs((metrics.heights.get(key) ?? -1) - h) > 0.5) {
            metrics.heights.set(key, h)
            changed = true
          }
        }
        if (changed) {
          rebuildOffsets(metrics, order)
          setWindowed(windowAt(metrics, el, atBottomRef.current))
        }
      }
    }
    lastKeyRef.current = lastKey
    lastSteeringIdRef.current = lastSteeringId
    followSigRef.current = followSig
    // Follow new flow content while pinned; do NOT re-pin on every render
    // merely because atBottomRef is true (scroll threshold → setState → snap).
    if (appendedUser || appendedSteering || (tipMoved && atBottomRef.current)) {
      toBottom(el)
      if (metricsRef.current !== null) setWindowed(windowAt(metricsRef.current, el, true))
    } else if (atBottomRef.current && metricsRef.current !== null) {
      // Pinned but the floor moved (streaming/history growth that did not
      // change the tip signature): re-pin and re-window to the new floor.
      const floor = Math.max(0, el.scrollHeight - el.clientHeight)
      if (floor - el.scrollTop > FOLLOW_THRESHOLD + 1) {
        toBottom(el)
        setWindowed(windowAt(metricsRef.current, el, true))
      }
    }
  })

  const onScrollRef = useRef(() => {})
  onScrollRef.current = () => {
    const local = listRef.current
    /* v8 ignore next -- ref-null guard: the handler only fires while mounted. */
    if (local === null) return
    const el = scrollerOf(local)
    // Only reader input may make raw scroll geometry change follow ownership:
    // a delivered position that deviates from the observed-top ledger (every
    // programmatic write records itself there synchronously). This covers
    // wheel, touch, scrollbar, and keyboard alike without naming devices.
    // Browser shrink-clamps land exactly on the floor min and delayed
    // programmatic deliveries land on the ledger itself, so both preserve
    // the current ownership state.
    const floor = Math.max(0, el.scrollHeight - el.clientHeight)
    const movedByReader = Math.abs(el.scrollTop - Math.min(observedTopRef.current, floor)) > 0.5
    const isAtBottom = movedByReader
      ? floor - el.scrollTop <= FOLLOW_THRESHOLD + 1
      : atBottomRef.current
    if (!movedByReader && isAtBottom) {
      toBottom(el)
      return
    }
    atBottomRef.current = isAtBottom
    setAtBottom(isAtBottom)
    const position = isAtBottom ? null : scrollPosition(local, el, metricsRef.current)
    if (isAtBottom) {
      anchorRef.current = null
    } else if (anchorRef.current !== null && position !== null) {
      anchorRef.current = { key: position.anchorKey, top: position.anchorTop }
    }
    // Continuous save (unmount happens after ref detach, so saving there is
    // too late); pinned-to-bottom clears so a remount keeps following.
    if (isAtBottom) chatScroll.save(null)
    else if (position !== null) chatScroll.save(position)
    observedTopRef.current = el.scrollTop
    // Windowed rendering: keep the mounted slice in sync with the scroll
    // position. Binary search + spacer math only; the set bails out when the
    // slice is unchanged, so steady scrolling costs one cheap computation per
    // event instead of a DOM pass. While pinned to the bottom the window
    // targets the live floor rather than the (possibly stale) scrollTop.
    const metrics = metricsRef.current
    if (metrics !== null) {
      const winTop = atBottomRef.current ? Math.max(0, el.scrollHeight - el.clientHeight) : el.scrollTop
      const next = computeWindow(winTop, el.clientHeight, order, metrics)
      setWindowed(prev => prev !== null
        && prev.start === next.start && prev.end === next.end
        && prev.topPad === next.topPad && prev.bottomPad === next.bottomPad
        ? prev : next)
    }
  }

  // Bind the scroll listener on the resolved scrollport once per mount;
  // reader-input attribution rides the observed-top ledger, not per-device
  // input listeners.
  useEffect(() => {
    const local = listRef.current
    /* v8 ignore next -- ref-null guard: effect runs after the list node commits. */
    if (local === null) return
    const el = scrollerOf(local)
    const onScroll = (): void => { onScrollRef.current() }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      el.removeEventListener('scroll', onScroll)
    }
  }, [])

  // The ref starts null and is assigned every render, so the placeholder
  // initializer a function initial value would need never exists.
  const followRef = useRef<(() => void) | null>(null)
  followRef.current = () => {
    const local = listRef.current
    if (local !== null && atBottomRef.current) {
      const el = scrollerOf(local)
      el.scrollTop = el.scrollHeight
      observedTopRef.current = el.scrollTop
      chatScroll.save(null)
    }
  }
  // Streaming, tool disclosures, and other flow changes resize the column;
  // the sticky composer resizes outside it. This observer owns ChatView's
  // dynamic-height follow decisions and writes only while the reader is pinned.
  useEffect(() => {
    const column = columnRef.current
    const local = listRef.current
    if (column === null || local === null || typeof ResizeObserver === 'undefined') return
    const scrollport = scrollerOf(local)
    const composer = scrollport.querySelector<HTMLElement>('[data-composer-seat]')
    const observer = new ResizeObserver(() => { followRef.current?.() })
    observer.observe(column)
    if (composer !== null) observer.observe(composer)
    return () => { observer.disconnect() }
  }, [])

  // A failed/empty page leaves the head unchanged. Once the request leaves
  // its busy state there is no future prepend for the saved anchor to own.
  useEffect(() => {
    if (!loadingOlder) anchorRef.current = null
  }, [loadingOlder])

  const loadOlderAnchored = (): void => {
    const local = listRef.current
    /* v8 ignore next -- ref-null guard: the paging button renders inside the list tree. */
    if (local !== null) {
      const el = scrollerOf(local)
      const anchor = pagingAnchor(local, el, metricsRef.current)
      if (anchor !== null) {
        if (anchor instanceof HTMLElement) {
          const key = anchor.dataset.chatAnchorKey
          if (key !== undefined) anchorRef.current = { key, top: flowTop(anchor, el) }
        } else {
          anchorRef.current = { key: anchor.key, top: anchor.top }
        }
      }
    }
    loadOlder()
  }

  const seat = (nodeKey: string): ReactNode => (
    <ChatNodeSeat
      key={nodeKey}
      nodeKey={nodeKey}
      useSession={useSession}
      selectedCallId={selectedCallId}
      cwd={cwd}
      openFile={requestOpenFile}
      inspectCall={inspectCall}
      forkAt={forkAt}
      renderMessageImages={renderMessageImages}
      fileMentions={fileMentions}
      renderSlot={renderSlot}
      t={t}
    />
  )

  return (
    <div className={css.root}>
      <div ref={listRef} className={css.scroll}>
        <div ref={columnRef} className={css.column} data-chat-flow="">
          {openState === 'loading' && <div className={css.hint}>{t('chat.loadingHistory')}</div>}
          {openState === 'error' && openError !== null && (
            <div className={css.openError}>
              {t('chat.loadError', { message: openError.message, code: openError.code })}
            </div>
          )}
          {hasMore && (
            <div className={css.older}>
              <button type="button" disabled={loadingOlder} onClick={loadOlderAnchored}>
                {loadingOlder ? t('loading') : t('chat.loadOlder')}
              </button>
            </div>
          )}
          {renderWindow === null ? (
            order.map(nodeKey => seat(nodeKey))
          ) : (
            <>
              {renderWindow.topPad > 0 && <div aria-hidden className={css.pad} style={{ height: renderWindow.topPad }} />}
              {order.slice(renderWindow.start, renderWindow.end).map(nodeKey => seat(nodeKey))}
              {renderWindow.bottomPad > 0 && <div aria-hidden className={css.pad} style={{ height: renderWindow.bottomPad }} />}
            </>
          )}
          {/* No pending placeholders: questions (ui-user-questions) and approvals
              (ApprovalPanel) both take over the composer, so a flow card would
              double-render the same wait. */}
          {/* Turn-level loading signal: rides the whole running turn (first-token
              wait, tool execution, streaming) so it never flickers per step. */}
          {running && <TurnStatus startTime={runningTurnStart} t={t} />}
          {pendingSteering.map(item => (
            <PendingSteeringBubble
              key={item.id}
              content={item.content}
              renderMessageImages={renderMessageImages}
              t={t}
            />
          ))}
        </div>
        {!atBottom && (
          <div className={css.toBottomSlot}>
            <button
              type="button"
              className={css.toBottom}
              aria-label={t('chat.toBottom')}
              onClick={() => {
                const local = listRef.current
                /* v8 ignore next -- ref-null guard: the button only renders alongside the mounted list. */
                if (local !== null) toBottom(scrollerOf(local))
              }}
            >
              <IconChevronDownOutline14 />
            </button>
          </div>
        )}
      </div>
      {fileOpenError !== null && (
        <FileOpenErrorDialog
          path={fileOpenError.path}
          message={fileOpenError.message}
          busy={fileOpenBusy}
          onClose={closeFileOpenError}
          onRetry={() => { requestOpenFile(fileOpenError.path) }}
          t={t}
        />
      )}
    </div>
  )
}

/** In-page Host open-path refusal: the wire reason plus a retry of the same path. */
function FileOpenErrorDialog({
  path, message, busy, onClose, onRetry, t,
}: {
  path: string
  message: string
  busy: boolean
  onClose: () => void
  onRetry: () => void
  t: ChatViewSlotProps['t']
}) {
  return (
    <Modal
      open
      onClose={onClose}
      closeLabel={t('close')}
      title={t(isFolderOpenPath(path) ? 'fileOpen.folderTitle' : 'fileOpen.title')}
      description={message}
      footer={(
        <>
          <Button variant="outline" className={css.modalAction} onClick={onClose}>{t('cancel')}</Button>
          <Button variant="primary" className={css.modalAction} disabled={busy} onClick={onRetry}>{t('retry')}</Button>
        </>
      )}
    />
  )
}
