#!/usr/bin/env bash
set -u

APP_DIR="/home/jetson/src/maoarm"
PYENV_ROOT="$HOME/.pyenv"
PYENV_ENV="maoarm"

WIFI_NAME="RoArm-M2"

MOD1="arm.control"
MOD2="cv"

TERM1_TITLE="RoboCat Control Terminal"
TERM2_TITLE="RoboCat Camera Terminal"

GUI1_TITLE="Figure 1"
GUI2_TITLE="Video"

get_screen_size() {
    xrandr | awk '/\*/ {split($1,a,"x"); print a[1], a[2]; exit}'
}

wait_for_window() {
    local title="$1"
    local tries="${2:-60}"

    while (( tries-- > 0 )); do
        if wmctrl -l | grep -F "$title" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

place_window() {
    local title="$1"
    local x="$2"
    local y="$3"
    local w="$4"
    local h="$5"

    wmctrl -r "$title" -b remove,maximized_vert,maximized_horz 2>/dev/null || true
    sleep 0.2
    wmctrl -r "$title" -e "0,${x},${y},${w},${h}"
}

startup_prompt() {
    local status_file
    status_file="$(mktemp)"

    xterm -T "RoboCat Startup" -geometry 60x10+200+200 -e bash -lc "
        WIFI_NAME='$WIFI_NAME'
        STATUS_FILE='$status_file'

        wifi_ready=0
        countdown=10

        while true; do
            clear
            echo

            if [[ \$wifi_ready -eq 0 ]]; then
                echo 'Waiting for Wi-Fi connection...'
                echo
                echo \"Required network: \$WIFI_NAME\"
                echo
                echo 'Press any key to cancel startup.'

                if nmcli -t -f NAME connection show --active | grep -Fxq \"\$WIFI_NAME\"; then
                    wifi_ready=1
                    continue
                fi
            else
                echo 'Wi-Fi connected.'
                echo
                echo 'I am about to wake up now!'
                echo 'Press any key within 10 seconds to let me sleep.'
                echo
                echo \"Starting in \${countdown}s...\"

                if [[ \$countdown -le 0 ]]; then
                    echo proceed > \"\$STATUS_FILE\"
                    exit 0
                fi

                countdown=\$((countdown - 1))
            fi

            if read -r -s -n 1 -t 1; then
                echo cancel > \"\$STATUS_FILE\"
                exit 0
            fi
        done
    "

    [[ -f "$status_file" ]] || return 1
    local result
    result="$(cat "$status_file")"
    rm -f "$status_file"

    [[ "$result" == "proceed" ]]
}

run_module_in_terminal() {
    local title="$1"
    local module="$2"

    xterm \
        -hold \
        -T "$title" \
        -geometry 100x30 \
        -e bash -lc "
            export PYENV_ROOT='$PYENV_ROOT'
            export PATH=\"\$PYENV_ROOT/bin:\$PATH\"

            cd '$APP_DIR' || exit 1

            eval \"\$(pyenv init - bash)\"
            PYENV_VERSION='$PYENV_ENV' pyenv exec python -m '$module'
        " &
}

sleep 5

if ! startup_prompt; then
    exit 0
fi

read -r SCREEN_W SCREEN_H < <(get_screen_size)
HALF_W=$((SCREEN_W / 2))
HALF_H=$((SCREEN_H / 2))

WIN_W=$(( HALF_W * 90 / 100 ))
WIN_H=$(( HALF_H * 90 / 100 ))

TOP_Y=0
BOTTOM_Y=$(( SCREEN_H - WIN_H ))
LEFT_X=0
RIGHT_X=$(( SCREEN_W - WIN_W ))

# run the arm module
run_module_in_terminal "$TERM1_TITLE" "$MOD1"

if wait_for_window "$TERM1_TITLE"; then
    place_window "$TERM1_TITLE" "$LEFT_X" "$BOTTOM_Y" "$WIN_W" "$WIN_H"
fi

# wait for arm/mood gui in background and place top-left
(
    if wait_for_window "$GUI1_TITLE" 120; then
        place_window "$GUI1_TITLE" "$LEFT_X" "$TOP_Y" "$WIN_W" "$WIN_H"
    fi
) &

sleep 5

# run camera module
run_module_in_terminal "$TERM2_TITLE" "$MOD2"

if wait_for_window "$TERM2_TITLE"; then
    place_window "$TERM2_TITLE" "$RIGHT_X" "$BOTTOM_Y" "$WIN_W" "$WIN_H"
fi

# wait for camera picture in background and place top-right
(
    if wait_for_window "$GUI2_TITLE" 120; then
        place_window "$GUI2_TITLE" "$RIGHT_X" "$TOP_Y" "$WIN_W" "$WIN_H"
    fi
) &
