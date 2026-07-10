#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

RUNTIME_ROOT=${ETPR1_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/etpr1_habitat}
SRC_ROOT=${RUNTIME_ROOT}/src
BUILD_ROOT=${RUNTIME_ROOT}/build
PREFIX=${RUNTIME_ROOT}/prefix
MANIFEST_ROOT=${RUNTIME_ROOT}/manifests
SITE_PACKAGES=${PREFIX}/site-packages
HABITAT_BASELINES_PREFIX=${PREFIX}/habitat-baselines
HABITAT_LAB_SOURCE=${ETPR1_HABITAT_LAB_SOURCE:-/home/a6000/gwl/_deps/habitat-lab-v0.3.3}
HABITAT_SIM_SOURCE=${ETPR1_HABITAT_SIM_SOURCE:-/home/a6000/gwl/_deps/habitat-sim-v0.3.3}
HABITAT_LAB_TARGET=${SRC_ROOT}/habitat-lab
HABITAT_SIM_TARGET=${SRC_ROOT}/habitat-sim
HABITAT_LAB_MARKER=${HABITAT_LAB_TARGET}/.etpr1_source_origin
HABITAT_SIM_MARKER=${HABITAT_SIM_TARGET}/.etpr1_source_origin

mkdir -p \
    "${SRC_ROOT}" \
    "${BUILD_ROOT}" \
    "${PREFIX}" \
    "${SITE_PACKAGES}" \
    "${PREFIX}/lib" \
    "${HABITAT_BASELINES_PREFIX}" \
    "${MANIFEST_ROOT}"

if [ ! -d "${HABITAT_LAB_SOURCE}" ]; then
    echo "Missing habitat-lab source directory: ${HABITAT_LAB_SOURCE}" >&2
    exit 1
fi

if [ ! -d "${HABITAT_SIM_SOURCE}" ]; then
    echo "Missing habitat-sim source directory: ${HABITAT_SIM_SOURCE}" >&2
    exit 1
fi

habitat_lab_status=reused
ensure_source_tree() {
    local name=$1
    local requested_source=$2
    local target=$3
    local marker=$4
    local status=reused
    local actual_local_source=MARKER_MISSING
    local marker_status=missing

    if [ ! -d "${target}" ]; then
        cp -a "${requested_source}" "${target}"
        printf '%s\n' "${requested_source}" > "${marker}"
        status=copied
        actual_local_source=${requested_source}
        marker_status=present
    elif [ -f "${marker}" ]; then
        actual_local_source=$(cat "${marker}")
        marker_status=present
        if [ "${actual_local_source}" != "${requested_source}" ]; then
            echo "Existing ${name} source tree at ${target} was created from ${actual_local_source}, not requested ${requested_source}. Remove the existing tree or point ETPR1_${name^^}_SOURCE back to the recorded origin." >&2
            status=source_mismatch
            printf '%s_status=%s\n' "${name}" "${status}"
            printf '%s_requested_source=%s\n' "${name}" "${requested_source}"
            printf '%s_actual_local_source=%s\n' "${name}" "${actual_local_source}"
            printf '%s_origin_marker_status=%s\n' "${name}" "${marker_status}"
            printf '%s_origin_marker_path=%s\n' "${name}" "${marker}"
            return 1
        fi
    else
        echo "Existing ${name} source tree at ${target} is missing provenance marker ${marker}. Remove the tree and rebuild it from a known source." >&2
        status=unknown_origin
        printf '%s_status=%s\n' "${name}" "${status}"
        printf '%s_requested_source=%s\n' "${name}" "${requested_source}"
        printf '%s_actual_local_source=%s\n' "${name}" "${actual_local_source}"
        printf '%s_origin_marker_status=%s\n' "${name}" "${marker_status}"
        printf '%s_origin_marker_path=%s\n' "${name}" "${marker}"
        return 1
    fi

    printf '%s_status=%s\n' "${name}" "${status}"
    printf '%s_requested_source=%s\n' "${name}" "${requested_source}"
    printf '%s_actual_local_source=%s\n' "${name}" "${actual_local_source}"
    printf '%s_origin_marker_status=%s\n' "${name}" "${marker_status}"
    printf '%s_origin_marker_path=%s\n' "${name}" "${marker}"
}

capture_provenance() {
    local output_file=$1
    shift

    if ! ensure_source_tree "$@" > "${output_file}"; then
        return 1
    fi
}

source_check_failed=0
habitat_lab_provenance_file=$(mktemp)
habitat_sim_provenance_file=$(mktemp)
trap 'rm -f "${habitat_lab_provenance_file}" "${habitat_sim_provenance_file}"' EXIT

if ! capture_provenance \
    "${habitat_lab_provenance_file}" \
    habitat-lab \
    "${HABITAT_LAB_SOURCE}" \
    "${HABITAT_LAB_TARGET}" \
    "${HABITAT_LAB_MARKER}"; then
    source_check_failed=1
fi

if ! capture_provenance \
    "${habitat_sim_provenance_file}" \
    habitat-sim \
    "${HABITAT_SIM_SOURCE}" \
    "${HABITAT_SIM_TARGET}" \
    "${HABITAT_SIM_MARKER}"; then
    source_check_failed=1
fi

habitat_lab_provenance=$(cat "${habitat_lab_provenance_file}")
habitat_sim_provenance=$(cat "${habitat_sim_provenance_file}")

if [ "${source_check_failed}" -ne 0 ]; then
    exit 1
fi

patch_habitat_lab_py311_compat() {
    local habitat_lab_root=$1

    HABITAT_LAB_PATCH_ROOT="${habitat_lab_root}" python - <<'PY'
import os
from pathlib import Path


def replace_once(text: str, old: str, new: str, file_path: Path) -> str:
    if old not in text:
        raise SystemExit(f"Expected patch target not found in {file_path}: {old}")
    return text.replace(old, new, 1)


root = Path(os.environ["HABITAT_LAB_PATCH_ROOT"])

patches = {
    root / "habitat-lab" / "habitat" / "config" / "default_structured_configs.py": [
        (
            "    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()\n",
            "    iterator_options: IteratorOptionsConfig = field(\n"
            "        default_factory=IteratorOptionsConfig\n"
            "    )\n",
        ),
        (
            "    fog_of_war: FogOfWarConfig = FogOfWarConfig()\n",
            "    fog_of_war: FogOfWarConfig = field(default_factory=FogOfWarConfig)\n",
        ),
        (
            "    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()\n",
            "    habitat_sim_v0: HabitatSimV0Config = field(default_factory=HabitatSimV0Config)\n",
        ),
        (
            "    renderer: RendererConfig = RendererConfig()\n",
            "    renderer: RendererConfig = field(default_factory=RendererConfig)\n",
        ),
        (
            "    environment: EnvironmentConfig = EnvironmentConfig()\n"
            "    simulator: SimulatorConfig = SimulatorConfig()\n",
            "    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)\n"
            "    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)\n",
        ),
        (
            "    gym: GymConfig = GymConfig()\n",
            "    gym: GymConfig = field(default_factory=GymConfig)\n",
        ),
    ],
    root / "habitat-lab" / "habitat" / "datasets" / "rearrange" / "run_episode_generator.py": [
        (
            "    params: SceneSamplerParamsConfig = SceneSamplerParamsConfig()\n",
            "    params: SceneSamplerParamsConfig = field(default_factory=SceneSamplerParamsConfig)\n",
        ),
        (
            "    scene_sampler: SceneSamplerConfig = SceneSamplerConfig()\n",
            "    scene_sampler: SceneSamplerConfig = field(default_factory=SceneSamplerConfig)\n",
        ),
    ],
    root / "habitat-lab" / "habitat" / "gym" / "gym_definitions.py": [
        (
            "def _try_register(id_name, entry_point, kwargs):\n"
            "    if id_name in registry.env_specs:\n",
            "def _try_register(id_name, entry_point, kwargs):\n"
            "    registry_envs = registry.env_specs if hasattr(registry, \"env_specs\") else registry\n"
            "    if id_name in registry_envs:\n",
        ),
        (
            "if \"Habitat-v0\" not in registry.env_specs:\n",
            "_registry_envs = registry.env_specs if hasattr(registry, \"env_specs\") else registry\n"
            "\n"
            "if \"Habitat-v0\" not in _registry_envs:\n",
        ),
    ],
    root / "habitat-baselines" / "habitat_baselines" / "config" / "default_structured_configs.py": [
        (
            "    action_dist: ActionDistributionConfig = ActionDistributionConfig()\n",
            "    action_dist: ActionDistributionConfig = field(default_factory=ActionDistributionConfig)\n",
        ),
        (
            "    agent: AgentAccessMgrConfig = AgentAccessMgrConfig()\n"
            "    preemption: PreemptionConfig = PreemptionConfig()\n",
            "    agent: AgentAccessMgrConfig = field(default_factory=AgentAccessMgrConfig)\n"
            "    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)\n",
        ),
        (
            "    ppo: PPOConfig = PPOConfig()\n"
            "    ddppo: DDPPOConfig = DDPPOConfig()\n"
            "    ver: VERConfig = VERConfig()\n",
            "    ppo: PPOConfig = field(default_factory=PPOConfig)\n"
            "    ddppo: DDPPOConfig = field(default_factory=DDPPOConfig)\n"
            "    ver: VERConfig = field(default_factory=VERConfig)\n",
        ),
        (
            "    vector_env_factory: VectorEnvFactoryConfig = VectorEnvFactoryConfig()\n"
            "    evaluator: EvaluatorConfig = EvaluatorConfig()\n",
            "    vector_env_factory: VectorEnvFactoryConfig = field(default_factory=VectorEnvFactoryConfig)\n"
            "    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)\n",
        ),
        (
            "    wb: WBConfig = WBConfig()\n",
            "    wb: WBConfig = field(default_factory=WBConfig)\n",
        ),
        (
            "    eval: EvalConfig = EvalConfig()\n"
            "    profiling: ProfilingConfig = ProfilingConfig()\n",
            "    eval: EvalConfig = field(default_factory=EvalConfig)\n"
            "    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)\n",
        ),
        (
            "    rl: RLConfig = RLConfig()\n",
            "    rl: RLConfig = field(default_factory=RLConfig)\n",
        ),
    ],
}

for file_path, replacements in patches.items():
    if not file_path.is_file():
        continue
    text = file_path.read_text(encoding="utf-8")
    for old, new in replacements:
        if new in text:
            continue
        text = replace_once(text, old, new, file_path)
    file_path.write_text(text, encoding="utf-8")
PY
}

patch_habitat_lab_py311_compat "${HABITAT_LAB_TARGET}"

python_cache_tag=$(
    python - <<'PY'
import sys

print(sys.implementation.cache_tag)
PY
)

sync_link() {
    local source_path=$1
    local dest_path=$2

    rm -rf "${dest_path}"
    ln -s "${source_path}" "${dest_path}"
}

sync_link_file() {
    local source_path=$1
    local dest_path=$2

    rm -f "${dest_path}"
    ln -s "${source_path}" "${dest_path}"
}

purge_stale_habitat_sim_build_dir() {
    local existing_binding
    existing_binding=$(find "${HABITAT_SIM_TARGET}/build" -type f -name "habitat_sim_bindings*.so" -print -quit 2>/dev/null || true)
    if [ -n "${existing_binding}" ]; then
        if ! ldd "${existing_binding}" 2>&1 | grep -q "not found"; then
            return
        fi
        rm -rf "${HABITAT_SIM_TARGET}/build"
        return
    fi

    local cache_file=${HABITAT_SIM_TARGET}/build/CMakeCache.txt
    if [ ! -f "${cache_file}" ]; then
        return
    fi

    local home_dir
    home_dir=$(
        sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${cache_file}" | head -n 1
    )
    local expected_home=${HABITAT_SIM_TARGET}/src
    if [ -z "${home_dir}" ] || [ "${home_dir}" != "${expected_home}" ]; then
        rm -rf "${HABITAT_SIM_TARGET}/build"
    fi
}

purge_stale_habitat_sim_build_dir

stage_compatible_habitat_sim_build() {
    if [ "${HABITAT_SIM_SOURCE}" = "${HABITAT_SIM_TARGET}" ]; then
        return
    fi

    local target_binding
    target_binding=$(find "${HABITAT_SIM_TARGET}/build" -type f -name "habitat_sim_bindings*.so" -print -quit 2>/dev/null || true)
    if [ -n "${target_binding}" ] && ! ldd "${target_binding}" 2>&1 | grep -q "not found"; then
        return
    fi

    local source_build=${HABITAT_SIM_SOURCE}/build
    local source_binding
    source_binding=$(
        find \
            "${source_build}/lib.linux-x86_64-${python_cache_tag}" \
            -type f \
            -name "habitat_sim_bindings*.so" \
            -print \
            -quit 2>/dev/null || true
    )
    if [ -z "${source_binding}" ]; then
        return
    fi
    if ldd "${source_binding}" 2>&1 | grep -q "not found"; then
        echo "Ignoring incompatible Habitat-Sim source binding: ${source_binding}" >&2
        return
    fi

    echo "Copying validated Habitat-Sim build from: ${source_build}"
    rm -rf "${HABITAT_SIM_TARGET}/build"
    cp -a "${source_build}" "${HABITAT_SIM_TARGET}/build"
}

stage_compatible_habitat_sim_build

build_habitat_sim_if_needed() {
    local binding
    binding=$(find "${HABITAT_SIM_TARGET}/build" -type f -name "habitat_sim_bindings*.so" -print -quit 2>/dev/null || true)
    if [ -n "${binding}" ]; then
        echo "Reusing Habitat-Sim native binding: ${binding}"
        return
    fi

    echo "Building Habitat-Sim native binding for $(python --version 2>&1)"
    (
        cd "${HABITAT_SIM_TARGET}"
        python setup.py \
            build_ext \
            --build-temp build \
            --parallel "${ETPR1_HABITAT_SIM_JOBS:-4}" \
            --headless \
            --no-update-submodules \
            --skip-install-magnum
    )
}

build_habitat_sim_if_needed

find_first_package_dir() {
    local package_name=$1
    shift

    local root
    for root in "$@"; do
        if [ -z "${root}" ] || [ ! -d "${root}" ]; then
            continue
        fi

        local package_dir
        while IFS= read -r package_dir; do
            if [ -f "${package_dir}/__init__.py" ]; then
                printf '%s\n' "${package_dir}"
                return 0
            fi
        done < <(find "${root}" -type d -path "*/${package_name}" | sort)
    done

    return 1
}

find_first_matching_file() {
    local pattern=$1
    shift

    local root
    for root in "$@"; do
        if [ -z "${root}" ] || [ ! -d "${root}" ]; then
            continue
        fi

        local file_path
        while IFS= read -r file_path; do
            if [ -f "${file_path}" ]; then
                printf '%s\n' "${file_path}"
                return 0
            fi
        done < <(find "${root}" -maxdepth 1 -type f -name "${pattern}" | sort)
    done

    return 1
}

find_parent_dir_for_matching_file() {
    local pattern=$1
    shift

    local root
    for root in "$@"; do
        if [ -z "${root}" ] || [ ! -d "${root}" ]; then
            continue
        fi

        local file_path
        while IFS= read -r file_path; do
            if [ -f "${file_path}" ]; then
                dirname "${file_path}"
                return 0
            fi
        done < <(find "${root}" -type f -name "${pattern}" | sort)
    done

    return 1
}

find_first_existing_dir() {
    local dir_path
    for dir_path in "$@"; do
        if [ -n "${dir_path}" ] && [ -d "${dir_path}" ]; then
            printf '%s\n' "${dir_path}"
            return 0
        fi
    done

    return 1
}

habitat_sim_package_root=$(
    find_first_package_dir \
        habitat_sim \
        "${HABITAT_SIM_TARGET}/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build" \
        "${HABITAT_SIM_TARGET}/src_python" || true
)

habitat_sim_extension_dir=$(
    find_parent_dir_for_matching_file \
        "habitat_sim_bindings*.so" \
        "${HABITAT_SIM_TARGET}/build/lib.linux-x86_64-${python_cache_tag}/habitat_sim/_ext" \
        "${HABITAT_SIM_TARGET}/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build" || true
)

magnum_package_root=$(
    find_first_package_dir \
        magnum \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python" \
        "${HABITAT_SIM_TARGET}/src/deps/magnum-bindings/src/python" || true
)

corrade_package_root=$(
    find_first_package_dir \
        corrade \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python" \
        "${HABITAT_SIM_TARGET}/src/deps/magnum-bindings/src/python" || true
)

corrade_extension_root=$(
    find_first_matching_file \
        "_corrade*.so" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python" || true
)

magnum_extension_root=$(
    find_first_matching_file \
        "_magnum*.so" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python/build/lib.linux-x86_64-${python_cache_tag}" \
        "${HABITAT_SIM_TARGET}/build/deps/magnum-bindings/src/python" || true
)

habitat_package_root=$(
    find_first_existing_dir \
        "${HABITAT_LAB_TARGET}/habitat-lab/habitat" \
        "${HABITAT_LAB_TARGET}/habitat" || true
)

habitat_baselines_package_root=$(
    find_first_existing_dir \
        "${HABITAT_LAB_TARGET}/habitat-baselines/habitat_baselines" \
        "${HABITAT_LAB_TARGET}/habitat_baselines" || true
)

if [ -n "${habitat_package_root}" ]; then
    sync_link "${habitat_package_root}" "${SITE_PACKAGES}/habitat"
fi

if [ -n "${habitat_baselines_package_root}" ]; then
    sync_link \
        "${habitat_baselines_package_root}" \
        "${HABITAT_BASELINES_PREFIX}/habitat_baselines"
fi

if [ -n "${habitat_sim_package_root}" ]; then
    sync_link "${habitat_sim_package_root}" "${SITE_PACKAGES}/habitat_sim"
fi

if [ -n "${habitat_sim_package_root}" ] && [ -n "${habitat_sim_extension_dir}" ]; then
    if [ "${habitat_sim_extension_dir}" != "${habitat_sim_package_root}/_ext" ]; then
        rm -rf "${habitat_sim_package_root}/_ext"
        ln -s "${habitat_sim_extension_dir}" "${habitat_sim_package_root}/_ext"
    fi
fi

if [ -n "${magnum_package_root}" ]; then
    sync_link "${magnum_package_root}" "${SITE_PACKAGES}/magnum"
fi

if [ -n "${corrade_package_root}" ]; then
    sync_link "${corrade_package_root}" "${SITE_PACKAGES}/corrade"
fi

if [ -n "${corrade_extension_root}" ]; then
    sync_link_file \
        "${corrade_extension_root}" \
        "${SITE_PACKAGES}/$(basename "${corrade_extension_root}")"
fi

if [ -n "${magnum_extension_root}" ]; then
    sync_link_file \
        "${magnum_extension_root}" \
        "${SITE_PACKAGES}/$(basename "${magnum_extension_root}")"
fi

cat > "${MANIFEST_ROOT}/source-roots.txt" <<EOF
${habitat_lab_provenance}
${habitat_sim_provenance}
EOF

cat > "${MANIFEST_ROOT}/staged-layout.txt" <<EOF
prefix=${PREFIX}
site_packages=${SITE_PACKAGES}
habitat=${SITE_PACKAGES}/habitat
habitat_source=${habitat_package_root:-MISSING}
habitat_baselines=${HABITAT_BASELINES_PREFIX}/habitat_baselines
habitat_baselines_source=${habitat_baselines_package_root:-MISSING}
habitat_sim=${SITE_PACKAGES}/habitat_sim
habitat_sim_source=${habitat_sim_package_root:-MISSING}
habitat_sim_ext=${SITE_PACKAGES}/habitat_sim/_ext
habitat_sim_ext_source=${habitat_sim_extension_dir:-MISSING}
magnum=${SITE_PACKAGES}/magnum
magnum_source=${magnum_package_root:-MISSING}
corrade=${SITE_PACKAGES}/corrade
corrade_source=${corrade_package_root:-MISSING}
_corrade=${SITE_PACKAGES}/$(basename "${corrade_extension_root:-_corrade_missing.so}")
_corrade_source=${corrade_extension_root:-MISSING}
_magnum=${SITE_PACKAGES}/$(basename "${magnum_extension_root:-_magnum_missing.so}")
_magnum_source=${magnum_extension_root:-MISSING}
EOF

echo "Using habitat-lab source: ${HABITAT_LAB_SOURCE}"
echo "Using habitat-sim source: ${HABITAT_SIM_SOURCE}"
echo "Source provenance:"
cat "${MANIFEST_ROOT}/source-roots.txt"
echo "Staged ETP-R1 runtime layout:"
cat "${MANIFEST_ROOT}/staged-layout.txt"

runtime_versions_tmp=$(mktemp)
if ! \
    ETPR1_EXPECT_HABITAT="${SITE_PACKAGES}/habitat" \
    ETPR1_EXPECT_HABITAT_SIM="${SITE_PACKAGES}/habitat_sim" \
    ETPR1_EXPECT_HABITAT_BASELINES="${HABITAT_BASELINES_PREFIX}/habitat_baselines" \
    python - > "${runtime_versions_tmp}" <<'PY'
import ast
import os
import sys
from pathlib import Path


CHECKS = (
    ("habitat", os.environ["ETPR1_EXPECT_HABITAT"]),
    ("habitat_sim", os.environ["ETPR1_EXPECT_HABITAT_SIM"]),
    ("habitat_baselines", os.environ["ETPR1_EXPECT_HABITAT_BASELINES"]),
)


def _read_version_from_file(version_file: Path):
    tree = ast.parse(version_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"__version__", "VERSION"}:
                return ast.literal_eval(node.value)
    return None


def _record_strict_version(name, expected_root_str):
    expected_path = Path(os.path.abspath(expected_root_str))

    if name in {"habitat", "habitat_baselines"}:
        version_candidates = [
            expected_path / "version.py",
            expected_path / "__init__.py",
        ]
    else:
        version_candidates = [expected_path / "__init__.py"]

    version = None
    checked_files = []
    for version_file in version_candidates:
        checked_files.append(str(version_file))
        if not version_file.is_file():
            continue
        try:
            version = _read_version_from_file(version_file)
        except Exception as exc:
            print(
                f"Failed to read staged {name} version from {version_file}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if version not in (None, ""):
            break

    if version in (None, ""):
        print(
            f"Staged {name} is missing __version__ in {', '.join(checked_files)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"{name}={version}")


for name, expected_root in CHECKS:
    _record_strict_version(name, expected_root)
PY
then
    rm -f "${runtime_versions_tmp}"
    exit 1
fi

mv "${runtime_versions_tmp}" "${MANIFEST_ROOT}/runtime-versions.txt"

echo "Staged ETP-R1 runtime versions:"
cat "${MANIFEST_ROOT}/runtime-versions.txt"

python - <<'PY' > "${MANIFEST_ROOT}/shared-core.txt"
import importlib


def _record_version(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{exc}")
        return

    version = getattr(module, "__version__", None)
    if version is None:
        print(f"{name}=missing")
        return

    print(f"{name}={version}")


for name in ("torch", "torchvision", "numpy", "transformers", "timm"):
    _record_version(name)
PY

echo "ETP-R1 runtime root prepared at ${RUNTIME_ROOT}."
