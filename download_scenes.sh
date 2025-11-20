#!/bin/bash

set -e

mkdir -p scenes

echo "Downloading fireplace_room scene..."
wget -O /tmp/fireplace_room.zip https://casual-effects.com/g3d/data10/research/model/fireplace_room/fireplace_room.zip
echo "Extracting to scenes/fireplace_room..."
unzip -o /tmp/fireplace_room.zip -d scenes/
echo "Cleaning up temp files..."
rm /tmp/fireplace_room.zip
