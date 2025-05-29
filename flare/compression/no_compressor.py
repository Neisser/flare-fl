from .compressors import BytesLike, Compressor


class NoCompression(Compressor):
    """A compressor that does no compression (passthrough)."""
    def compress(self, data: BytesLike) -> BytesLike:
        print("NoCompression: compress called (passthrough).")
        return data

    def decompress(self, data: BytesLike) -> BytesLike:
        print("NoCompression: decompress called (passthrough).")
        return data
