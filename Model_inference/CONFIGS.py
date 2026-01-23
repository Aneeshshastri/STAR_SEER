class Config:
    # --- Paths ---
    H5_PATH = "/kaggle/input/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5" 
    TFREC_DIR = "/kaggle/working/tfrecords"
    STATS_PATH = "/kaggle/working/dataset_stats.npz"
    
    # --- System ---
    TESTING_MODE = True
    TEST_LIMIT = 10000 
    NUM_SHARDS = 16 
    TRAIN_INTEGRATED=True
    TRAIN_BASE_MODEL=True
    # --- Model Hyperparameters ---
    BATCH_SIZE = 512       
    LEARNING_RATE = 1e-3  
    EPOCHS = 50
    LATENT_DIM = 268
    OUTPUT_LENGTH = 8575
    # --loss related---
    L2_VAL = 1e-4          
    INPUT_NOISE = 0.05     
    IVAR_SCALE = 1000.0   
    CLIP_NORM = 1.0     
    BADPIX_CUTOFF=1e-3  
    #----predictor-labels--------
    #CAUTION: LITERALLY EVERYTHING IS IN THE SAME ORDER AS THESE LABELS. DO NOT TOUCH THE ORDER OF THESE LABELS
    SELECTED_LABELS = [
        # 1. Core
        'TEFF', 'LOGG', 'FE_H', 'VMICRO', 'VMACRO', 'VSINI',
        # 2. CNO
        'C_FE', 'N_FE', 'O_FE',
        #3. metals
        'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
        'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE','K_FE',
        'NA_FE','V_FE','CO_FE'
    ]
    ABUNDANCE_INDICES =[]#[i for i, label in enumerate(SELECTED_LABELS) if '_FE' in label]
    FE_H_INDEX = SELECTED_LABELS.index('FE_H')
    N_LABELS = len(SELECTED_LABELS) + 1
    #GRAPHING:
    WAVELENGTH_START = 1514
    WAVELENGTH_END = 1694 