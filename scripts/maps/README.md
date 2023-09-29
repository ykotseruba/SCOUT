## Creating maps for DR(eye)VE and BDD-A

Install dependencies:

```
pip3 install -r requirements.txt
```

Pull the latest valhalla docker image (if anything goes wrong at this step, see https://github.com/gis-ops/docker-valhalla for up-to-date instructions)

```
docker pull ghcr.io/gis-ops/docker-valhalla/valhalla:latest
```

Run from the valhalla_scripts folder

For DReyeVE run docker with OSM tiles for north-east Italy:

```
./run_valhalla_docker_DReyeVE.sh
```

For BDD-A run docker with OSM tiles for Southern california:

```
./run_valhalla_docker_BDD-A.sh
```

Once the docker is up and running (it might take a while), use the `process_gps.py` script to generate street networks and map-matched route for every video in BDD-A and DReyeVE that have a valid GPS data.

```
python3 process_gps.py --dataset <dataset_name> --vid_id <video_id>
```

Options:

- dataset: DReyeVE or BDD-A
- vid-id: number of the video or -1 to process all videos 