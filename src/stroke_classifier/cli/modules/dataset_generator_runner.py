import argschema

from stroke_classifier.cli.schemas.dataset_generator import \
    DatasetGeneratorSchema
from stroke_classifier.tile_classifier.dataset_generator import \
    DatasetGenerator


class DatasetGeneratorRunner(argschema.ArgSchemaParser):
    default_schema = DatasetGeneratorSchema

    def run(self):
        dataset_generator = DatasetGenerator(
            tile_width=self.args['tile_width'],
            tile_height=self.args['tile_height'],
            fg_thresh=self.args['foreground_threshold'],
            data_dir=self.args['data_dir']
        )
        dataset_generator.get_tiles(
            meta_path=self.args['meta_path'],
            out_path=self.args['out_path']
        )


if __name__ == '__main__':
    DatasetGeneratorRunner().run()
