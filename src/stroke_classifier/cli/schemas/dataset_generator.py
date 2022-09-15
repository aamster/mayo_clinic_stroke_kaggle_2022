import argschema


class DatasetGeneratorSchema(argschema.ArgSchema):
    tile_width = argschema.fields.Int(
        default=224,
        description='tile width'
    )
    tile_height = argschema.fields.Int(
        default=224,
        description='tile height'
    )
    foreground_threshold = argschema.fields.Float(
        default=0.1,
        description='foreground threshold'
    )
    ignore_image_ids = argschema.fields.List(
        argschema.fields.Str,
        required=False,
        help='List of known bad image ids',
        cli_as_single_argument=True
    )
    meta_path = argschema.fields.InputFile(
        required=True,
        help='Path to meta csv'
    )
    data_dir = argschema.fields.InputDir(
        required=True,
        help='path to data'
    )
    out_path = argschema.fields.OutputFile(
        required=True,
        help='where to save dataset'
    )
