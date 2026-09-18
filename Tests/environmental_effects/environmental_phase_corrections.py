import numpy as np
import pytest
import warnings
import pyEFPEHM


def base_parameters():
    """Simple vacuum binary configuration for testing the new code"""
    # define binary parameters
    return {
        "mass1": 10.0,
        "mass2": 5.0,
        "eccentricity": 0.1,
        "f22_start": 0.01,
        "f22_end": 0.1,
        "distance": 100.0,
        "harmonic_array": [[2, 2, 2],[2, 2, 3],],

        "environmental_model": None,
        "environmental_redshift": 0.0,
        "environmental_phase_sign": 1.0,

        "tertiary_mass": 0.0,
        "tertiary_distance": 0.0,
        "gas_density": 0.0,
        "sound_speed": 0.0,
    }


def make_waveform(**updates):
    """Construct a waveform after updating the base parameters."""
    # load the base test parameters
    params = base_parameters()
    # apply the requested parameter updates
    params.update(updates)
    # return the constructed waveform
    return pyEFPEHM.pyEFPE(params)

# --------- 1. Test vacuum recovery ---------
    
def test_environmental_model_none_recovers_vacuum():
    """Test whether the default model and the explicit 'environmental_model=None' produce identical waveforms"""
    # create evenly spaced test frequencies
    freqs = np.linspace(0.01, 0.10, 128)

    # load the base test parameters
    params_default = base_parameters()
    # remove the parameter to test its default behavior
    params_default.pop("environmental_model")

    # load the base test parameters
    params_explicit_none = base_parameters()
    # set the environmental model explicitly to none
    params_explicit_none["environmental_model"] = None

    # initialize pyEFPEHM waveform model using the two sets of parameters
    waveform_default = pyEFPEHM.pyEFPE(params_default)
    waveform_none = pyEFPEHM.pyEFPE(params_explicit_none)

    # generate waveforms with the two sets of parameters
    h_default = waveform_default.generate_waveform(freqs)
    h_none = waveform_none.generate_waveform(freqs)

    # verify the values match within tolerance
    np.testing.assert_allclose(h_none, h_default, rtol=0.0, atol=0.0)


def test_no_environment_returns_zero_phase():
    """Test whether the environmental helper returns zero with 'environmental_model=None' """
    
    # generate a waveform with a specific environment 
    waveform = make_waveform(environmental_model=None)

    # define frequency and eccentricity values
    f22 = np.array([20.0, 30.0, 40.0])
    e = np.array([0.1, 0.08, 0.05])

    # calculate the environmental phase correction
    correction = waveform._delta_psi2_environment(f22, e)

    # verify that the arrays match exactly
    np.testing.assert_array_equal(correction,np.zeros_like(f22))


# --------- 2. Test every environmental effect ---------

def test_roemer_ell2_scalings():
    """Test whether the Roemer delay function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""

    # generate a waveform with Roemer delay
    waveform = make_waveform(
        environmental_model="roemer",
        tertiary_mass=1.0e5,
        tertiary_distance=1.495978707e11,  # 1 AU
    )

    # define eccentricity and frequency arrays
    e = np.array([0.1])
    f = np.array([0.01])

    # calculate the environmental phase correction
    phase_1 = waveform._delta_psi2_roemer(f, e)

    # verify the arrays have equal shape and finite and positive values
    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # calculate the environmental phase correction for 2*f
    phase_2f = waveform._delta_psi2_roemer(2.0 * f, e)

    # verify the values match the frequency scaling: f^(-13/3) within tolerance
    np.testing.assert_allclose(phase_2f / phase_1, 2.0**(-13.0 / 3.0), rtol=1.0e-13)

    # tertiary-mass scaling: proportional to m3
    # update the environmental parameter for scaling
    original_mass = waveform.params["tertiary_mass"]
    # update the environmental parameter for scaling
    waveform.params["tertiary_mass"] = 2.0 * original_mass

    # calculate the environmental phase correction
    phase_2m = waveform._delta_psi2_roemer(f, e)

    # verify the values match within tolerance
    np.testing.assert_allclose(phase_2m / phase_1, 2.0, rtol=1.0e-13)

    # restore the parameter
    # update the environmental parameter for scaling
    waveform.params["tertiary_mass"] = original_mass

    # separation scaling: proportional to R^(-2)
    # update the environmental parameter for scaling
    original_distance = waveform.params["tertiary_distance"]
    waveform.params["tertiary_distance"] = 2.0 * original_distance

    # calculate the environmental phase correction
    phase_2R = waveform._delta_psi2_roemer(f, e)

    # verify the values match within tolerance
    np.testing.assert_allclose(phase_2R / phase_1, 0.25, rtol=1.0e-13)


def test_bhl_ell2_scalings():
    """Test whether the Bondi-Hoyle-Littleton drag function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""

    # generate waveform with BHL drag
    waveform = make_waveform(
        environmental_model="bhl",
        gas_density=1.0e-7,
        sound_speed=1.0e4,
    )

    # define eccentricity and frequency arrays
    e = np.array([0.1])
    f = np.array([0.01])

    # calculate the environmental phase correction
    phase_1 = waveform._delta_psi2_bhl(f, e)

    # verify the arrays have equal shape and finite and positive values
    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # calculate the environmental phase correction for 2*f
    phase_2f = waveform._delta_psi2_bhl(2.0 * f, e)

    # verify the values match the frequency scaling: f^(-14/3) within tolerance
    np.testing.assert_allclose(phase_2f / phase_1, 2.0**(-14.0 / 3.0), rtol=1.0e-13)

    # update the environmental parameter for scaling
    original_density = waveform.params["gas_density"]
    waveform.params["gas_density"] = 2.0 * original_density

    # calculate the environmental phase correction
    phase_2rho = waveform._delta_psi2_bhl(f, e)

    # density scaling: proportional to rho
    # verify the values match within tolerance
    np.testing.assert_allclose(phase_2rho / phase_1, 2.0, rtol=1.0e-13)

    # update the environmental parameter
    waveform.params["gas_density"] = original_density

    # sound-speed scaling: proportional to cs^(-2)
    # update the environmental parameter
    original_sound_speed = waveform.params["sound_speed"]
    waveform.params["sound_speed"] = 2.0 * original_sound_speed

    # compute the environmental phase correction
    phase_2cs = waveform._delta_psi2_bhl(f, e)

    # verify the values match within tolerance
    np.testing.assert_allclose(phase_2cs / phase_1, 0.25, rtol=1.0e-13)


def test_supersonic_ell2_scalings():
    """Test whether the supersonic drag function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""
    
    # generate waveform with supersonic drag
    waveform = make_waveform(
        environmental_model="supersonic",
        gas_density=1.0e-7,
    )

    # define eccentricity and frequency arrays
    e = np.array([0.1])
    f = np.array([0.01])

    # compute waveform phase
    phase_1 = waveform._delta_psi2_supersonic(f, e)

    # verify the arrays have equal shape and finite and positive values
    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # compute waveform phase with 2*f
    phase_2f = waveform._delta_psi2_supersonic(2.0 * f, e)

    # verify the values match the frequency scaling: f^(-16/3)
    np.testing.assert_allclose(phase_2f / phase_1, 2.0**(-16.0 / 3.0), rtol=1.0e-13)
    
    # update the gas density for scaling
    original_density = waveform.params["gas_density"]
    waveform.params["gas_density"] = 2.0 * original_density

    # compute waveform phase with 2*rho
    phase_2rho = waveform._delta_psi2_supersonic(f, e)

    # verify the values match the density scaling (proportional to rho)
    np.testing.assert_allclose(phase_2rho / phase_1, 2.0, rtol=1.0e-13)

    
# --------- 3. Test mapping from dominant eccentric harmonic to higher harmonics ---------

# parameterize the test with multiple cases
@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)

def test_mapping_to_higher_eccentric_harmonics(model, extra_parameters):

    # generate a waveform with a specific environment
    waveform = make_waveform(environmental_model=model, **extra_parameters)

    # define frequency, eccentricity and ell arrays
    f_detector = np.array([0.01, 0.05, 0.1])
    e = np.array([0.1, 0.1, 0.1])
    ell = np.array([2, 3, 5])

    # compute phase correction
    calculated = waveform._environmental_harmonic_phase(f_detector, e, ell)

    # map detector frequencies to equivalent dominant mode frequencies
    f22_equivalent = 2.0 * f_detector / ell

    # calculate the expected harmonic phase
    expected = (0.5 * ell * waveform._delta_psi2_environment(f22_equivalent, e))

    # verify the values match within tolerance
    np.testing.assert_allclose(calculated, expected, rtol=1.0e-13, atol=0.0)

def test_ell2_mapping_recovers_base_phase():

    # generate a waveform with BHL drag
    waveform = make_waveform(environmental_model="bhl", gas_density=1.0e-7, sound_speed=1.0e4)

    # define frequency, eccentricity and ell arrays
    f = np.array([0.01, 0.05, 0.1])
    e = np.array([0.1, 0.08, 0.05])
    ell = np.full(f.shape, 2)

    # calculate the environmental phase correction for the dominant mode
    harmonic_phase = waveform._environmental_harmonic_phase(f, e, ell)

    # calculate the environmental phase correction 
    base_phase = waveform._delta_psi2_environment(f, e)

    # verify the values match within tolerance
    np.testing.assert_allclose(harmonic_phase, base_phase, rtol=1.0e-13, atol=0.0)

    
def test_zero_harmonic_receives_no_correction():
    
    # generate a waveform with BHL drag
    waveform = make_waveform(environmental_model="bhl", gas_density=1.0e-7, sound_speed=1.0e4)

    # define frequency, eccentricity and ell arrays
    f = np.array([20.0, 30.0])
    e = np.array([0.1, 0.1])
    ell = np.array([0, 0])

    # calculate the environmental phase correction for the ell = 0 mode
    correction = waveform._environmental_harmonic_phase(f, e, ell)

    # verify the values match exactly
    np.testing.assert_array_equal(correction, np.zeros_like(f))


# --------- 4. Test complete waveform generation ---------

# parameterize the test with multiple cases
@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)

def test_complete_environmental_waveform_generation(model, extra_parameters):
    """Test all three models run through 'stationary_times()', harmonic correction, SUA amplitudes, and mode summation without errors or NANs"""
    
    # create evenly spaced frequencies
    freqs = np.linspace(0.01, 0.1, 128)

    # generate a waveform
    waveform = make_waveform(environmental_model=model, **extra_parameters)

    # compute GW polarizations
    hplus, hcross = waveform.generate_waveform(freqs)

    # verify that the arrays have the same shape and have finite values
    assert hplus.shape == freqs.shape
    assert hcross.shape == freqs.shape
    assert np.all(np.isfinite(hplus))
    assert np.all(np.isfinite(hcross))

    # verify that the waveform contains at least some nonzero samples
    assert np.any(np.abs(hplus) + np.abs(hcross) > 0.0)


@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)

def test_complete_mode_generation(model, extra_parameters):
    """Test whether 'generate_modes()' produces corrected 'psi_SPA' arrays"""
    
    # create evenly spaced frequencies
    freqs = np.linspace(0.01, 0.1, 128)

    # generate a waveform
    waveform = make_waveform(environmental_model=model, **extra_parameters)

    # generate waveform modes
    result = waveform.generate_modes(freqs, return_waveform_pieces=True)

    # verify that result includes at least one mode
    assert len(result["modes"]) > 0

    # iterate through the generated modes and verify that values are finite
    for mode_label, mode in result["modes"].items():
        assert np.all(np.isfinite(mode["polarizations"]))
        assert np.all(np.isfinite(mode["psi_SPA"]))
        assert np.all(np.isfinite(mode["t_SPA"]))
        assert np.all(np.isfinite(mode["T_SPA"]))

def test_tdomain_warns_when_environment_is_selected():
    """Test whether the code returns a warning when trying to call a waveform with environmental effects in time domain"""

    # generate waveform with supersonic drag
    waveform = make_waveform(environmental_model="supersonic",gas_density=1.0)

    # call waveform in time domain
    with pytest.warns(UserWarning, match="Environmental phase corrections"):
        waveform.generate_tdomain_waveform(times=np.array([waveform.return_start_time()]))

def test_tdomain_does_not_warn_for_vacuum():
    """Verify that the vacuum waveform returns no warning related to environmental effects when called in time domain"""
    waveform = make_waveform(environmental_model=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        waveform.generate_tdomain_waveform(times=np.array([waveform.return_start_time()]))