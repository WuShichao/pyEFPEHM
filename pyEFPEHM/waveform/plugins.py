""" Plugins for external packages to access the pyEFPEHM waveform model(s)

Copyright (c) 2026 Gonzalo Morras (gonzalo.morras@aei.mpg.de)

This file is part of pyEFPEHM.
Licensed under the Apache License. See the LICENSE file in the project root for details.
"""

import numpy as np

# No imports should be placed here outside those required for the pyEFPEHM
# package itself to avoid increasing any install requirements for the
# model.

def rename_parameters(parameters):
    # Parameter name conversions, all others passed directly
    # This helps keep a consistent naming convention set for downstream
    # Where possible, sticking to
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py
    #
    # Note that the native names may be provided in most cases in lieue of
    # these, except in the case of 'f_lower'.
    conversions = {'f_lower': 'f22_start',
                   'f_ref': 'f22_ref',
                   'f_final': 'f22_end',
                   'coa_phase': 'phase',
                   'mean_per_ano': 'mean_anomaly',
                   'dquad_mon1': 'q1',
                   'dquad_mon2': 'q2',
                   }

    renamed_params = {
        new_name: parameters[old_name]
        for old_name, new_name in conversions.items()
        if (old_name in parameters) and (parameters[old_name] is not None)
    }
    parameters.update(renamed_params)


def _mode_array_to_pyEFPEHM(parameters):
    """ Convert a PyCBC-style mode_array (list of (l, m) or (l, m, n) tuples, where n is the
    eccentric harmonic index) into pyEFPEHM's own mode_array/harmonic_array parameters.

    (l, m) entries select the whole (l, m) co-precessing multipole, letting pyEFPEHM choose which
    eccentric harmonics n to include dynamically (controlled by Amplitude_tol); (l, m, n) entries
    pin down one specific harmonic via pyEFPEHM's `harmonic_array`.

    IMPORTANT: pyEFPEHM's `harmonic_array` is all-or-nothing -- as soon as any (l, m, n) entry is
    given, EFPE.pyEFPE.__init__ derives the *entire* set of active modes from harmonic_array alone
    (see process_harmonic_array in amplitudes.py) and no longer does its own dynamic per-(l,m)
    harmonic selection. So a plain (l, m) entry alongside any (l, m, n) entry would silently be
    dropped instead of getting "all its harmonics" as one might expect from mixing the two. To
    avoid silently returning an incomplete waveform, we require mode_array to be either all (l, m)
    or all (l, m, n) -- not a mix. If you need both (some modes with dynamically-chosen harmonics,
    others pinned to one specific harmonic), call get_fd_waveform_modes/get_td_waveform_modes twice
    (once per kind) and combine the results yourself.
    """
    mode_array = parameters.pop('mode_array', None)
    if mode_array is None:
        return

    lengths = {len(entry) for entry in mode_array}
    if lengths - {2, 3}:
        raise ValueError("mode_array entries must be (l, m) or (l, m, n), got %s"%(mode_array,))
    if lengths == {2, 3}:
        raise ValueError("mode_array mixes plain (l, m) entries with (l, m, n) entries (%s); "
                         "pyEFPEHM's harmonic_array is all-or-nothing, so this cannot be honoured "
                         "in one call -- see _mode_array_to_pyEFPEHM's docstring."%(mode_array,))

    if lengths == {2}:
        parameters['mode_array'] = [list(entry) for entry in mode_array]
    else:
        harmonic_array = [list(entry) for entry in mode_array]
        parameters['harmonic_array'] = harmonic_array
        parameters['mode_array'] = [entry[:2] for entry in harmonic_array]


def pycbc_fd_plugin(**parameters):
    """ Interface for the PyCBC package: frequency-domain, summed [h_+, h_x]
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    wf = EFPE.pyEFPE(parameters)
    M = parameters['mass1'] + parameters['mass2']
    freqs = np.arange(parameters['f_lower'],
                      f_SchwarzISCO(M),
                      parameters['delta_f']
                      )

    hp, hc = wf.generate_waveform(freqs)

    epoch = wf.return_start_time()
    hp = FrequencySeries(hp, epoch=epoch, delta_f=parameters['delta_f'])
    hc = FrequencySeries(hc, epoch=epoch, delta_f=parameters['delta_f'])
    return hp, hc


def pycbc_td_plugin(**parameters):
    """ Interface for the PyCBC package: time-domain, summed [h_+, h_x]
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    wf = EFPE.pyEFPE(parameters)

    hp, hc = wf.generate_tdomain_waveform(delta_t=parameters["delta_t"])

    epoch = wf.return_start_time()
    hp = TimeSeries(hp, epoch=epoch, delta_t=parameters['delta_t'])
    hc = TimeSeries(hc, epoch=epoch, delta_t=parameters['delta_t'])
    return hp, hc


r""" Mode-by-mode plugins (feed PyCBC's get_fd_waveform_modes/get_td_waveform_modes)

pyEFPEHM natively decomposes the waveform by co-precessing-frame (l, m, n) harmonic (n is the
eccentric sub-harmonic index; see generate_hlm_modes/generate_tdomain_hlm_modes in EFPE.py). Each
such harmonic further contributes to *every* inertial-frame index mp=-l..l (this is what precession
"twist-up" means). Because two different co-precessing m's can independently produce a harmonic with
the same n that both contribute to the same target (l, mp), we key the returned dictionaries by the
4-tuple (l, mp, n, m) -- m being the *co-precessing* label the harmonic came from -- rather than
collapsing/summing same-(l, mp, n) contributions from different m together. Summing across m (and
across n, if desired) for a given (l, mp) is left to the caller.

CONVENTION: this 'hlm' is exactly the twist-up mode h^{lm'} of Eq.(3) of arXiv:2604.11903 (the pyEFPEHM
paper), which satisfies the textbook relation h_+ - i h_x = sum_{l,mp} h^{lm'} (-2)Y^{lm'}(theta, phi) of
Eq.(2) of that paper with NO conjugate -- i.e. the same convention used by e.g. LAL/PyCBC's own hlm
generators (see pycbc.waveform.waveform_modes.sum_modes). Concretely (see generate_hlm_modes and
generate_tdomain_hlm_modes docstrings in EFPE.py for the derivation):

    frequency domain:  [h_+, h_x](f) = sum_mp hlm[mp](f) * conj(Proj_{l,mp}(theta, phi))
    time domain:       [h_+, h_x](t) = Re[ sum_mp hlm[mp](t) * Proj_{l,mp}(theta, phi) ]

where Proj is built the same way as pyEFPE.pyEFPE.Apc_projs (see interpolate_Apc_prec in EFPE.py) at
the desired (theta, phi). The conj() in the FD case is *not* a departure from the textbook hlm
convention: it is purely a Fourier-transform sign-convention artifact (the paper's Eq.(16) uses a
+2*pi*i*f*t kernel, the opposite sign from the standard/LAL/PyCBC -2*pi*i*f*t kernel; since h(t) is
real, converting between the two is a single overall complex conjugate). No such conjugate is needed
in the TD case, since the paper's Eq.(2) is already stated directly in the time domain.
"""

def pycbc_fd_modes_plugin(**parameters):
    """ Interface for the PyCBC package: frequency-domain (l, mp, n, m) modes, not summed.

    Returns
    -------
    ulm : dict
        (l, mp, n, m) -> Re(hlm(f)), as a pycbc.types.FrequencySeries
    vlm : dict
        (l, mp, n, m) -> Im(hlm(f)), as a pycbc.types.FrequencySeries
    """
    from . import EFPE
    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    parameters['Compute_hlm_Modes'] = True
    wf = EFPE.pyEFPE(parameters)

    M = parameters['mass1'] + parameters['mass2']
    delta_f = parameters['delta_f']
    freqs = np.arange(parameters['f_lower'], f_SchwarzISCO(M), delta_f)

    result = wf.generate_hlm_modes(freqs)
    epoch = wf.return_start_time()

    ulm, vlm = {}, {}
    for (l, m, n), mode in result['modes'].items():
        for j, mp in enumerate(mode['mp']):
            key = (l, int(mp), int(n), int(m))
            hlm_full = np.zeros(len(freqs), dtype=np.complex128)
            hlm_full[mode['freq_idxs']] = mode['hlm'][:, j]
            ulm[key] = FrequencySeries(hlm_full.real, epoch=epoch, delta_f=delta_f)
            vlm[key] = FrequencySeries(hlm_full.imag, epoch=epoch, delta_f=delta_f)
    return ulm, vlm


def pycbc_td_modes_plugin(**parameters):
    """ Interface for the PyCBC package: time-domain (l, mp, n, m) modes, not summed.

    Returns
    -------
    ulm : dict
        (l, mp, n, m) -> Re(hlm(t)), as a pycbc.types.TimeSeries
    vlm : dict
        (l, mp, n, m) -> Im(hlm(t)), as a pycbc.types.TimeSeries
    """
    from . import EFPE
    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    parameters['Compute_hlm_Modes'] = True
    wf = EFPE.pyEFPE(parameters)

    delta_t = parameters['delta_t']
    result = wf.generate_tdomain_hlm_modes(delta_t=delta_t)
    times = result['times']

    epoch = wf.return_start_time()

    ulm, vlm = {}, {}
    for (l, m, n), mode in result['modes'].items():
        for j, mp in enumerate(mode['mp']):
            key = (l, int(mp), int(n), int(m))
            hlm_full = np.zeros(len(times), dtype=np.complex128)
            hlm_full[mode['time_idxs']] = mode['hlm'][:, j]
            ulm[key] = TimeSeries(hlm_full.real, epoch=epoch, delta_t=delta_t)
            vlm[key] = TimeSeries(hlm_full.imag, epoch=epoch, delta_t=delta_t)
    return ulm, vlm
