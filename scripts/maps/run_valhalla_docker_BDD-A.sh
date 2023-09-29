docker run -it --rm --name valhalla_bdda -p 8002:8002 \
-v $PWD/valhalla/custom_files:/custom_files \
-e tile_urls=https://download.geofabrik.de/north-america/us/california-latest.osm.pbf \
-e serve_tiles=True -e build_admins=True \
ghcr.io/gis-ops/docker-valhalla/valhalla:latest