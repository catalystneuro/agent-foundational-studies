#!/bin/bash

# Quick analysis runner - handles both complete and incomplete files

NWB_FILE="/var/folders/67/qdwczmzx315gj1xp7hp1f11r0000gn/T/dandi_nwb_yn0sovnu/session.nwb"
EXPECTED_SIZE=8417855886
TIMEOUT_SECONDS=3600  # 1 hour

echo "========================================="
echo "Theta Phase Entrainment Analysis"
echo "DANDI 000003: Hippocampal Recordings"
echo "========================================="
echo ""

# Check if file exists
if [ ! -f "$NWB_FILE" ]; then
    echo "ERROR: NWB file not found at $NWB_FILE"
    exit 1
fi

ACTUAL_SIZE=$(stat -f%z "$NWB_FILE")
PCT=$((ACTUAL_SIZE * 100 / EXPECTED_SIZE))

echo "File status:"
echo "  Path: $NWB_FILE"
echo "  Size: $((ACTUAL_SIZE / 1024 / 1024 / 1024)) GB / $((EXPECTED_SIZE / 1024 / 1024 / 1024)) GB ($PCT%)"
echo ""

# If file is >90% complete, run analysis
if [ $ACTUAL_SIZE -gt $((EXPECTED_SIZE * 90 / 100)) ]; then
    echo "File download is complete or nearly complete. Starting analysis..."
    python3 theta_phase_entrainment.py
elif [ $ACTUAL_SIZE -gt $((EXPECTED_SIZE / 2)) ]; then
    echo "File is $PCT% complete. Starting analysis (may handle partially downloaded file)..."
    python3 theta_phase_entrainment.py
else
    echo "Waiting for file to reach 50%+ complete before starting analysis..."
    echo "Current: $PCT%"
    echo ""
    echo "Monitoring download progress..."

    start_time=$(date +%s)
    while true; do
        ACTUAL_SIZE=$(stat -f%z "$NWB_FILE" 2>/dev/null || echo 0)
        PCT=$((ACTUAL_SIZE * 100 / EXPECTED_SIZE))
        elapsed=$(($(date +%s) - start_time))

        if [ $ACTUAL_SIZE -gt $((EXPECTED_SIZE / 2)) ]; then
            echo "File reached 50%+. Starting analysis..."
            python3 theta_phase_entrainment.py
            break
        fi

        if [ $elapsed -gt $TIMEOUT_SECONDS ]; then
            echo "Timeout waiting for file. Attempting analysis on partial file..."
            python3 theta_phase_entrainment.py
            break
        fi

        echo "$(date '+%H:%M:%S') Download progress: $PCT% ($((ACTUAL_SIZE/1024/1024/1024)) GB / $((EXPECTED_SIZE/1024/1024/1024)) GB)"
        sleep 120
    done
fi
