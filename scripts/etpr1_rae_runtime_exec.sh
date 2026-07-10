#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_ROOT=${ETPR1_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/etpr1_habitat}
RUNTIME_PREFIX=${ETPR1_RUNTIME_PREFIX:-${RUNTIME_ROOT}/prefix}
RUNTIME_SITE_PACKAGES=${ETPR1_RUNTIME_SITE_PACKAGES:-${RUNTIME_PREFIX}/site-packages}
RUNTIME_HABITAT_BASELINES=${ETPR1_RUNTIME_HABITAT_BASELINES:-${RUNTIME_PREFIX}/habitat-baselines}
RUNTIME_LIB=${ETPR1_RUNTIME_LIB:-${RUNTIME_PREFIX}/lib}
LEGACY_CLIP_ROOT=${ETPR1_LEGACY_CLIP_ROOT:-${REPO_ROOT}/vendor/legacy_clip}

fail() {
    echo "Invalid ETP-R1 runtime: $*" >&2
    exit 1
}

resolved_path() {
    readlink -f -- "$1" 2>/dev/null || return 1
}

reject_forbidden_path() {
    local path=$1
    local forbidden_etpnav="/ETP""Nav"
    case "${path}" in
        *"${forbidden_etpnav}"*|*"/dino_cwp"*|*"/_deps"*)
            fail "forbidden path: ${path}"
            ;;
    esac
}

require_owned_dir() {
    local path=$1
    local owner=$2
    local label=$3
    [ -d "${path}" ] || fail "missing ${label}: ${path}"

    local resolved owner_resolved
    resolved=$(resolved_path "${path}") || fail "cannot resolve ${label}: ${path}"
    owner_resolved=$(resolved_path "${owner}") || fail "cannot resolve owner for ${label}: ${owner}"
    reject_forbidden_path "${resolved}"
    case "${resolved}" in
        "${owner_resolved}"|"${owner_resolved}"/*) ;;
        *) fail "${label} escapes ETP-R1 owner ${owner_resolved}: ${resolved}" ;;
    esac
}

require_owned_file() {
    local path=$1
    local owner=$2
    local label=$3
    [ -f "${path}" ] || fail "missing ${label}: ${path}"

    local resolved owner_resolved
    resolved=$(resolved_path "${path}") || fail "cannot resolve ${label}: ${path}"
    owner_resolved=$(resolved_path "${owner}") || fail "cannot resolve owner for ${label}: ${owner}"
    reject_forbidden_path "${resolved}"
    case "${resolved}" in
        "${owner_resolved}"/*) ;;
        *) fail "${label} escapes ETP-R1 owner ${owner_resolved}: ${resolved}" ;;
    esac
}

require_owned_dir "${REPO_ROOT}/.runtime" "${REPO_ROOT}" "project runtime directory"
require_owned_dir "${RUNTIME_ROOT}" "${REPO_ROOT}/.runtime" "runtime root"
require_owned_dir "${RUNTIME_PREFIX}" "${RUNTIME_ROOT}" "runtime prefix"
require_owned_dir "${RUNTIME_SITE_PACKAGES}/habitat" "${RUNTIME_ROOT}" "habitat package"
require_owned_dir "${RUNTIME_SITE_PACKAGES}/habitat_sim" "${RUNTIME_ROOT}" "habitat_sim package"
require_owned_dir \
    "${RUNTIME_HABITAT_BASELINES}/habitat_baselines" \
    "${RUNTIME_ROOT}" \
    "habitat_baselines package"
require_owned_dir "${RUNTIME_LIB}" "${RUNTIME_ROOT}" "runtime library directory"
require_owned_dir "${LEGACY_CLIP_ROOT}" "${REPO_ROOT}/vendor" "legacy CLIP root"
require_owned_dir "${LEGACY_CLIP_ROOT}/clip" "${LEGACY_CLIP_ROOT}" "legacy CLIP package"
require_owned_file "${LEGACY_CLIP_ROOT}/clip/__init__.py" "${LEGACY_CLIP_ROOT}" "legacy CLIP initializer"

shopt -s nullglob
habitat_sim_bindings=("${RUNTIME_SITE_PACKAGES}/habitat_sim/_ext"/habitat_sim_bindings*.so)
corrade_bindings=("${RUNTIME_SITE_PACKAGES}"/_corrade*.so)
magnum_bindings=("${RUNTIME_SITE_PACKAGES}"/_magnum*.so)
shopt -u nullglob

[ "${#habitat_sim_bindings[@]}" -gt 0 ] || fail "missing Habitat-Sim native binding"
[ "${#corrade_bindings[@]}" -gt 0 ] || fail "missing Corrade native binding"
[ "${#magnum_bindings[@]}" -gt 0 ] || fail "missing Magnum native binding"

for binding in "${habitat_sim_bindings[@]}"; do
    require_owned_file "${binding}" "${RUNTIME_ROOT}" "Habitat-Sim native binding"
done
for binding in "${corrade_bindings[@]}"; do
    require_owned_file "${binding}" "${RUNTIME_ROOT}" "Corrade native binding"
done
for binding in "${magnum_bindings[@]}"; do
    require_owned_file "${binding}" "${RUNTIME_ROOT}" "Magnum native binding"
done

export ETPR1_RUNTIME_ACTIVE=1
export ETPR1_RUNTIME_ROOT="${RUNTIME_ROOT}"
export ETPR1_RUNTIME_PREFIX="${RUNTIME_PREFIX}"
export ETPR1_LEGACY_CLIP_ROOT="${LEGACY_CLIP_ROOT}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
# Habitat-Sim's EGL path can fail if conda's libGLdispatch is loaded together
# with the host NVIDIA GL stack. Prefer the system dispatcher when present.
SYSTEM_GL_DISPATCH=${ETPR1_SYSTEM_GL_DISPATCH:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}
if [ -f "${SYSTEM_GL_DISPATCH}" ]; then
    case " ${LD_PRELOAD:-} " in
        *" ${SYSTEM_GL_DISPATCH} "*) ;;
        *) export LD_PRELOAD="${SYSTEM_GL_DISPATCH}${LD_PRELOAD:+ ${LD_PRELOAD}}" ;;
    esac
fi
export PYTHONPATH="${REPO_ROOT}:${LEGACY_CLIP_ROOT}:${RUNTIME_SITE_PACKAGES}:${RUNTIME_HABITAT_BASELINES}"
export LD_LIBRARY_PATH="${RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "$@"
