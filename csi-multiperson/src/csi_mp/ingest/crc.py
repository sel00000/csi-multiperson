"""Legacy CSI wire CRC: CCITT-FALSE, poly 0x1021, init 0xffff."""

from binascii import crc_hqx


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """Return the non-reflected CCITT-FALSE checksum (no final XOR)."""
    if not 0 <= crc <= 0xFFFF:
        raise ValueError("crc must be a uint16")
    return crc_hqx(data, crc)


crc16_ccitt_false = crc16_ccitt
