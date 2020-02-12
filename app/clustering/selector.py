def get_clustering(config, loader, output_dir):
    if config.type == 'brown':
        from app.clustering.brown import BrownClustering
        return BrownClustering(loader, output_dir, config.clusters)
    else:
        raise Exception(f'Unknown clustering: {config.type}')
