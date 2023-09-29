docker run -it --rm --name valhalla_dreyeve -p 8002:8002 \
-v $PWD/valhalla/custom_files:/custom_files \
-e tile_urls=http://download.geofabrik.de/europe/italy/nord-est-latest.osm.pbf \
-e serve_tiles=True -e build_admins=True \
ghcr.io/gis-ops/docker-valhalla/valhalla:latest