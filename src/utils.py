import binascii
import hashlib
import struct


def bytes_to_crc_32_int(b):
    return binascii.crc32(b)


def bytes_to_sha_256_int(b):
    hash_obj = hashlib.sha256(b)
    return struct.unpack("<I", hash_obj.digest()[-4:])[0]
