#!/usr/bin/env bash

config_scalar() {
    local config_file=$1
    local section=$2
    local field=$3
    awk -v wanted_section="$section" -v wanted_field="$field" '
        $0 ~ "^[[:space:]]*" wanted_section ":[[:space:]]*$" {
            in_section=1
            next
        }
        in_section && /^[^[:space:]#]/ { in_section=0 }
        in_section && $0 ~ "^[[:space:]]+" wanted_field ":[[:space:]]*" {
            sub("^[[:space:]]+" wanted_field ":[[:space:]]*", "")
            sub(/[[:space:]]+#.*/, "")
            gsub(/^[[:space:]"'\'' ]+|[[:space:]"'\'' ]+$/, "")
            print $0
            exit
        }
    ' "$config_file"
}

checkpoint_order_from_config() {
    local config_file=$1 value
    value=$(config_scalar "$config_file" EVAL checkpoint_order)
    value=${value,,}
    printf '%s\n' "${value:-descending}"
}

checkpoint_watch_dir_from_config() {
    local config_file=$1 destination
    destination=$(config_scalar \
        "$config_file" IL checkpoint_sync_destination)
    if [[ "$destination" == *:/* ]]; then
        destination=${destination#*:}
    fi
    printf '%s\n' "$destination"
}

validate_checkpoint_order() {
    case "$1" in
        ascending|descending) ;;
        *)
            echo "checkpoint_order must be ascending or descending, got: $1" >&2
            return 2
            ;;
    esac
}

list_ordered_checkpoints() {
    local checkpoint_dir=$1
    local order=$2
    validate_checkpoint_order "$order" || return
    if [ "$order" = ascending ]; then
        find "$checkpoint_dir" -maxdepth 1 -type f \
            -name 'ckpt.iter*.pth' | sort -V
    else
        find "$checkpoint_dir" -maxdepth 1 -type f \
            -name 'ckpt.iter*.pth' | sort -Vr
    fi
}
