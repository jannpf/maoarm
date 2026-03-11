#!/usr/bin/env bash
set -u

APP_DIR="/home/jetson/src/maoarm"
PYENV_ROOT="$HOME/.pyenv"
PYENV_ENV="maoarm"

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

show_cancel_prompt() {
    local status_file
    status_file="$(mktemp)"

    xterm \
        -T "RoboCat Startup" \
        -geometry 60x8+200+200 \
        -fa Monospace \
        -fs 14 \
        -e bash -lc "
            clear
            echo
            echo 'I am about to wake up now!.'
            echo 'Press any key within 10 seconds to let me sleep.'
            echo

            for ((i=10; i>=1; i--)); do
                echo -ne \"\rStarting in \${i}s... \"
                if read -r -s -n 1 -t 1; then
                    echo
                    echo
                    echo 'Startup cancelled.'
                    echo cancel > '$status_file'
                    sleep 1
                    exit 0
                fi
            done

            echo
            echo proceed > '$status_file'
            exit 0
        "

    local result="cancel"
    if [[ -f "$status_file" ]]; then
        result="$(cat "$status_file")"
        rm -f "$status_file"
    fi

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

if ! show_cancel_prompt; then
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
