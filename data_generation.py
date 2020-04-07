
import sys
import numpy as np

def construct_generator(
    molecules,
    PPMdir,
    Ymode                = 'D-S-H',
    batch_size           = 30,
    nBestRotations       = 20,
    scan_dim             = (128,128,30),
    nRot                 = 100,
    pixPerAngstrome      = 5,
    Qs                   = [-0.1],
    iZPPs                = [8],
    tip_type             = 'quadrupole',
    tipZs                = None,
    tipQs                = None,
    rot_tolerance        = 5,
    Yrange               = 2,
    distAbove            = 7.5,
    distAboveDelta       = None,
    molCentering         = 'box',
    Rpp                  = -0.5,
    dzmax                = 1.2,
    dzmax_s              = 1.2,
    diskMode             = 'sphere',
    zmin                 = -1.5,
    zmin_xyz             = -2.0,
    Nmax_xyz             = 30,
    shuffle_rotations    = True,
    shuffle_molecules    = True,
    preName              = '../Molecules/',
    postName             = '.xyz',
    maxTilt0             = 0.0,
    wz                   = 1.0,
    df_weight_steps      = 10,
    dz                   = 0.1,
    randomize_enabled    = True,
    randomize_parameters = False,
    randomize_tip_tilt   = True,
    randomize_distance   = True,
    rndQmax              = 0.1,
    rndRmax              = 0.2,
    rndEmax              = 0.5,
    rndAlphaMax          = 0.1,
    rotations            = None,
    Rfunc                = None,
    invStep              = None,
    Rmax                 = None,
    lvec = np.array([
        [ 0.0,  0.0,  0.0],
        [30.0,  0.0,  0.0],
        [ 0.0, 30.0,  0.0],
        [ 0.0,  0.0, 30.0]])):

    # ============ Setup Probe Particle

    if not isinstance(Qs, list):
        Qs = [Qs]
    if not isinstance(iZPPs, list):
        iZPPs = [iZPPs]

    sys.path.append(PPMdir)
    import pyopencl
    import pyProbeParticle.common           as PPU
    import pyProbeParticle.GeneratorOCL_LJC as PPGen
    import pyProbeParticle.oclUtils         as oclu 
    
    # Initialize OpenCL kernels and stuff
    try:
        env = oclu.OCLEnvironment( i_platform = 0 )
        PPGen.FFcl.init(env)
        PPGen.oclr.init(env)
    except pyopencl.cffi_cl.RuntimeError as e:
        print('Environment could not be initialized. Maybe already was.')
        print('Error:')
        print(e)

    if rotations is None:
        # Make rotations
        rotations = PPU.sphereTangentSpace(n=nRot) # http://blog.marmakoide.org/?p=1
    
    # Make data generator
    data_generator = PPGen.Generator( molecules, rotations, batch_size, pixPerAngstrome=pixPerAngstrome, Ymode=Ymode )

    # Handle tip types
    if tip_type == 'monopole':
        if tipZs is None:
            tipZs = [0,0,0,0]
        if tipQs is None:
            tipQs = [1,0,0,0]
    elif tip_type == 'dipole':
        if tipZs is None:
            tipZs = [0.1,-0.1,0,0]
        if tipQs is None:
            tipQs = [10,-10,0,0]
    elif tip_type == 'quadrupole':
        if tipZs is None:
            tipZs = [0.1,0,-0.1,0]
        if tipQs is None:
            tipQs = [100,-200,100,0]
    else:
        raise ValueError("Got tip_type %s. Should be 'monopole', 'dipole', or 'quadrupole'." % str(tip_type))

    # Set probe particle(s)
    tipZs = np.array(tipZs)
    tipQs = np.array(tipQs)
    data_generator.bQZ = True
    data_generator.iZPPs = iZPPs
    data_generator.QZs = [tipZs for _ in range(len(Qs))]
    data_generator.Qs = [q*tipQs for q in Qs]
    data_generator.rot_tolerance = rot_tolerance

    if data_generator.projector is not None:
        data_generator.projector.Rpp = Rpp
        data_generator.projector.dzmax = dzmax
        data_generator.projector.dzmax_s = dzmax_s
        data_generator.projector.zmin = zmin
        data_generator.projector.Rfunc = Rfunc
        data_generator.projector.invStep = invStep
        data_generator.projector.Rmax = Rmax
    
    data_generator.preName = preName
    data_generator.postName = postName
    data_generator.scan_dim = scan_dim
    data_generator.nBestRotations = nBestRotations
    data_generator.shuffle_rotations = shuffle_rotations
    data_generator.shuffle_molecules = shuffle_molecules
    data_generator.Yrange = Yrange
    data_generator.lvec = lvec
    data_generator.distAbove = distAbove
    data_generator.distAboveDelta = distAboveDelta
    data_generator.molCentering = molCentering
    data_generator.molCenterAfm = True
    data_generator.maxTilt0 = maxTilt0
    data_generator.zmin_xyz = zmin_xyz
    data_generator.Nmax_xyz = Nmax_xyz
    data_generator.diskMode = diskMode

    # --- 'MultiMapSpheres' settings  ztop[ichan] = (R - Rmin)/Rstep
    data_generator.nChan = 3      # number of channels, resp. atom size bins
    data_generator.Rmin  = 1.4    # minimum atom radius
    data_generator.Rstep = 0.2    # size range per bin (resp. channel)

    # Params randomization
    data_generator.randomize_enabled    = randomize_enabled
    data_generator.randomize_tip_tilt   = randomize_tip_tilt
    data_generator.randomize_parameters = randomize_parameters
    data_generator.randomize_distance   = randomize_distance
    data_generator.rndQmax              = rndQmax    # charge += rndQmax * ( rand()-0.5 )  (negative is off)
    data_generator.rndRmax              = rndRmax
    data_generator.rndEmax              = rndEmax
    data_generator.rndAlphaMax          = rndAlphaMax

    # z-weight exp(-wz*z)
    data_generator.wz      = wz
    data_generator.zWeight = data_generator.getZWeights();

    # Weight function for Fz -> df conversion ( oscillation amplitude 1.0Angstroem = 10 * 0.1 ( 10 n steps, dz=0.1 Angstroem step lenght ) )
    dfWeight = PPU.getDfWeight( df_weight_steps, dz=dz )[0].astype(np.float32)
    data_generator.dfWeight = dfWeight

    data_generator.initFF()

    return data_generator
