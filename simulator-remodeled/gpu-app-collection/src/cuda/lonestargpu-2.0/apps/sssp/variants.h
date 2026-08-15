#pragma once

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

#define SSSP_LS  0
#define SSSP_WLN 1 // worklist, Nasre
#define SSSP_WLC 2  // worklist, cub-based
#define SSSP_TPATM 3 // from topology-atomic.cu

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==SSSP_LS
#include "sssp_ls.h"
#elif VARIANT==SSSP_WLN
#include "sssp_worklistn.h"
#elif VARIANT==SSSP_WLC
#include "sssp_worklistc.h"
#elif VARIANT==SSSP_TPATM
#include "sssp_topoatomic.h"
#else 
#error "Unknown variant"
#endif
