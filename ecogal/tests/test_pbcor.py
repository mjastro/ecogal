import numpy as np

def test_metadata_query():
    from .. import pbcor

    meta = pbcor.get_pbcor_metadata()
    assert(meta["file_alma"] == "2022.1.01644.S__all_MOSDEF_3324_b3_cont_noninter2sig.image.pbcor.fits")

    assert(meta["band"] == 3)

    meta = pbcor.get_pbcor_metadata(file_alma="2019.1.01528.S__all_1mm.1_b4_cont_noninter2sig.image.pbcor.fits")
    assert(meta["file_alma"] == "2019.1.01528.S__all_1mm.1_b4_cont_noninter2sig.image.pbcor.fits")

    assert(meta["band"] == 4)


def test_pbcor_file():
    from .. import pbcor
    
    pbc = pbcor.EcogalFile(
        file_alma="2019.1.01528.S__all_1mm.1_b4_cont_noninter2sig.image.pbcor.fits"
    )
    
    assert pbc.shape == (1, 320, 320)

    assert np.allclose(np.nanmax(pbc.data_sn), 13.540124)

