import zlib

from .compressors import BytesLike, Compressor


class ZlibCompressor(Compressor):
    """Compressor using zlib."""
    def __init__(self, level: int = zlib.Z_DEFAULT_COMPRESSION):
        self.level = level
        print(f"ZlibCompressor initialized with level: {level}")

    def compress(self, data: BytesLike) -> BytesLike:
        print(f"ZlibCompressor: compressing {len(data)} bytes...")
        compressed_data = zlib.compress(data, self.level)
        print(f"ZlibCompressor: compressed to {len(compressed_data)} bytes.")
        return compressed_data

    def decompress(self, data: BytesLike) -> BytesLike:
        print(f"ZlibCompressor: decompressing {len(data)} bytes...")
        decompressed_data = zlib.decompress(data)
        print(f"ZlibCompressor: decompressed to {len(decompressed_data)} bytes.")
        return decompressed_data
