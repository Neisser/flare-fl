import gzip
import io

from .compressors import BytesLike, Compressor


class GzipCompressor(Compressor):
    """Compressor using gzip."""
    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel
        print(f"GzipCompressor initialized with compresslevel: {compresslevel}")

    def compress(self, data: BytesLike) -> BytesLike:
        print(f"GzipCompressor: compressing {len(data)} bytes...")
        out = io.BytesIO()
        with gzip.GzipFile(
            fileobj=out, mode='wb',
            compresslevel=self.compresslevel
        ) as f:
            f.write(data)
        compressed_data = out.getvalue()
        print(f"GzipCompressor: compressed to {len(compressed_data)} bytes.")
        return compressed_data

    def decompress(self, data: BytesLike) -> BytesLike:
        print(f"GzipCompressor: decompressing {len(data)} bytes...")
        in_ = io.BytesIO(data)
        with gzip.GzipFile(fileobj=in_, mode='rb') as f:
            decompressed_data = f.read()
        print(f"GzipCompressor: decompressed to {len(decompressed_data)} bytes.")
        return decompressed_data
