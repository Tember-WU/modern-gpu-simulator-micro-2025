#pragma once

// Shared trace-control state is defined here so every build variant, including
// legacy variants that do not use the limit, can be compiled by the aggregate
// Lonestar make target.
static bool lonestar_trace_iteration_limit_reached = false;

static int lonestar_trace_max_iterations()
{
	const char *value = getenv("LONESTAR_TRACE_MAX_ITERATIONS");
	if (value == NULL) return 0;
	const int limit = atoi(value);
	return limit > 0 ? limit : 0;
}

static bool lonestar_trace_skip_verification()
{
	const char *value = getenv("LONESTAR_TRACE_SKIP_VERIFY");
	return value != NULL && atoi(value) != 0;
}

#define BFS_LS 0
#define BFS_ATOMIC 1
#define BFS_MERRILL 2
#define BFS_WORKLISTW 3 // worklist version from worklist directory
#define BFS_WORKLISTG 4 // deleted experimental version
#define BFS_WORKLISTA 5  // bitmasks
#define BFS_WORKLISTC 6  // cub

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==BFS_LS
#include "bfs_ls.h"
#elif VARIANT==BFS_ATOMIC
#include "bfs_topo_atomic.h"
#elif VARIANT==BFS_MERRILL
#include "bfs_merrill.h"
#elif VARIANT==BFS_WORKLISTW
#include "bfs_worklistw.h"
#elif VARIANT==BFS_WORKLISTG
#include "bfs_worklistg.h"
#elif VARIANT==BFS_WORKLISTA
#include "bfs_worklista.h"
#elif VARIANT==BFS_WORKLISTC
#include "bfs_worklistc.h"
#else 
#error "Unknown variant"
#endif
