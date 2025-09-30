"""EMG feature extraction module.

This module provides various feature extraction methods for EMG signals
including time-domain, frequency-domain, and time-frequency features.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import List, Dict, Optional, Tuple
import warnings


class EMGFeatureExtractor:
    """
    Feature extractor for EMG signals.
    
    Extracts various types of features from EMG data:
    - Time-domain features (MAV, RMS, VAR, etc.)
    - Frequency-domain features (MNF, MDF, etc.)
    - Time-frequency features (wavelet coefficients)
    - Statistical features (skewness, kurtosis, etc.)
    """
    
    def __init__(self, sampling_rate: int = 1000):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling rate of EMG signals in Hz
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []
    
    def extract_time_domain_features(self, emg_data: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features from EMG data.
        
        Args:
            emg_data: EMG data of shape (channels, samples) or (samples,)
            
        Returns:
            Array of time-domain features
        """
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        
        features = []
        feature_names = []
        
        for ch in range(emg_data.shape[0]):
            signal_ch = emg_data[ch, :]
            
            # Mean Absolute Value (MAV)
            mav = np.mean(np.abs(signal_ch))
            features.append(mav)
            feature_names.append(f"MAV_ch{ch}")
            
            # Root Mean Square (RMS)
            rms = np.sqrt(np.mean(signal_ch**2))
            features.append(rms)
            feature_names.append(f"RMS_ch{ch}")
            
            # Variance (VAR)
            var = np.var(signal_ch)
            features.append(var)
            feature_names.append(f"VAR_ch{ch}")
            
            # Standard Deviation (STD)
            std = np.std(signal_ch)
            features.append(std)
            feature_names.append(f"STD_ch{ch}")
            
            # Waveform Length (WL)
            wl = np.sum(np.abs(np.diff(signal_ch)))
            features.append(wl)
            feature_names.append(f"WL_ch{ch}")
            
            # Zero Crossings (ZC)
            zc = self._count_zero_crossings(signal_ch)
            features.append(zc)
            feature_names.append(f"ZC_ch{ch}")
            
            # Slope Sign Changes (SSC)
            ssc = self._count_slope_sign_changes(signal_ch)
            features.append(ssc)
            feature_names.append(f"SSC_ch{ch}")
            
            # Willison Amplitude (WAMP)
            wamp = self._willison_amplitude(signal_ch, threshold=0.01)
            features.append(wamp)
            feature_names.append(f"WAMP_ch{ch}")
            
            # Skewness
            skewness = skew(signal_ch)
            features.append(skewness)
            feature_names.append(f"SKEW_ch{ch}")
            
            # Kurtosis
            kurt = kurtosis(signal_ch)
            features.append(kurt)
            feature_names.append(f"KURT_ch{ch}")
        
        self.feature_names.extend(feature_names)
        return np.array(features)
    
    def extract_frequency_domain_features(self, emg_data: np.ndarray) -> np.ndarray:
        """
        Extract frequency-domain features from EMG data.
        
        Args:
            emg_data: EMG data of shape (channels, samples) or (samples,)
            
        Returns:
            Array of frequency-domain features
        """
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        
        features = []
        feature_names = []
        
        for ch in range(emg_data.shape[0]):
            signal_ch = emg_data[ch, :]
            
            # Compute power spectral density
            freqs, psd = signal.welch(signal_ch, fs=self.sampling_rate, nperseg=min(256, len(signal_ch)//4))
            
            # Mean Frequency (MNF)
            mnf = np.sum(freqs * psd) / np.sum(psd)
            features.append(mnf)
            feature_names.append(f"MNF_ch{ch}")
            
            # Median Frequency (MDF)
            cumsum_psd = np.cumsum(psd)
            mdf_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0][0]
            mdf = freqs[mdf_idx]
            features.append(mdf)
            feature_names.append(f"MDF_ch{ch}")
            
            # Peak Frequency
            peak_freq = freqs[np.argmax(psd)]
            features.append(peak_freq)
            feature_names.append(f"PEAK_FREQ_ch{ch}")
            
            # Spectral Centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            features.append(spectral_centroid)
            feature_names.append(f"SPECTRAL_CENTROID_ch{ch}")
            
            # Spectral Rolloff (95%)
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            features.append(spectral_rolloff)
            feature_names.append(f"SPECTRAL_ROLLOFF_ch{ch}")
            
            # Band Power (0-50 Hz, 50-150 Hz, 150-250 Hz)
            bands = [(0, 50), (50, 150), (150, 250)]
            for low, high in bands:
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[band_mask])
                features.append(band_power)
                feature_names.append(f"BAND_POWER_{low}_{high}Hz_ch{ch}")
        
        self.feature_names.extend(feature_names)
        return np.array(features)
    
    def extract_time_frequency_features(self, emg_data: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
        """
        Extract time-frequency features using wavelet transform.
        
        Args:
            emg_data: EMG data of shape (channels, samples) or (samples,)
            wavelet: Wavelet type for decomposition
            
        Returns:
            Array of time-frequency features
        """
        try:
            import pywt
        except ImportError:
            warnings.warn("PyWavelets not available. Skipping time-frequency features.")
            return np.array([])
        
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        
        features = []
        feature_names = []
        
        for ch in range(emg_data.shape[0]):
            signal_ch = emg_data[ch, :]
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal_ch, wavelet, level=4)
            
            # Extract features from each level
            for level, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    # Energy
                    energy = np.sum(coeff**2)
                    features.append(energy)
                    feature_names.append(f"WAVELET_ENERGY_L{level}_ch{ch}")
                    
                    # Mean absolute value
                    mav = np.mean(np.abs(coeff))
                    features.append(mav)
                    feature_names.append(f"WAVELET_MAV_L{level}_ch{ch}")
                    
                    # Standard deviation
                    std = np.std(coeff)
                    features.append(std)
                    feature_names.append(f"WAVELET_STD_L{level}_ch{ch}")
        
        self.feature_names.extend(feature_names)
        return np.array(features)
    
    def extract_all_features(self, emg_data: np.ndarray, include_time_freq: bool = True) -> np.ndarray:
        """
        Extract all available features from EMG data.
        
        Args:
            emg_data: EMG data of shape (channels, samples) or (samples,)
            include_time_freq: Whether to include time-frequency features
            
        Returns:
            Array of all extracted features
        """
        self.feature_names = []
        
        # Time-domain features
        time_features = self.extract_time_domain_features(emg_data)
        
        # Frequency-domain features
        freq_features = self.extract_frequency_domain_features(emg_data)
        
        # Time-frequency features
        if include_time_freq:
            time_freq_features = self.extract_time_frequency_features(emg_data)
        else:
            time_freq_features = np.array([])
        
        # Combine all features
        all_features = np.concatenate([
            time_features,
            freq_features,
            time_freq_features
        ])
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get the names of extracted features."""
        return self.feature_names.copy()
    
    def _count_zero_crossings(self, signal: np.ndarray, threshold: float = 0.0) -> int:
        """Count zero crossings in the signal."""
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crossings)
    
    def _count_slope_sign_changes(self, signal: np.ndarray, threshold: float = 0.0) -> int:
        """Count slope sign changes in the signal."""
        diff_signal = np.diff(signal)
        sign_changes = np.where(np.diff(np.signbit(diff_signal)))[0]
        return len(sign_changes)
    
    def _willison_amplitude(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """Calculate Willison amplitude."""
        diff_signal = np.abs(np.diff(signal))
        wamp = np.sum(diff_signal > threshold)
        return wamp
    
    def extract_features_from_windows(
        self,
        emg_data: np.ndarray,
        window_size: int,
        overlap: float = 0.5,
        include_time_freq: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from sliding windows of EMG data.
        
        Args:
            emg_data: EMG data of shape (channels, samples) or (samples,)
            window_size: Size of each window in samples
            overlap: Overlap ratio between windows (0.0 to 1.0)
            include_time_freq: Whether to include time-frequency features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
            feature_matrix: Shape (num_windows, num_features)
            feature_names: List of feature names
        """
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        
        step_size = int(window_size * (1 - overlap))
        num_windows = (emg_data.shape[1] - window_size) // step_size + 1
        
        feature_matrix = []
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            window_data = emg_data[:, start_idx:end_idx]
            features = self.extract_all_features(window_data, include_time_freq)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        feature_names = self.get_feature_names()
        
        return feature_matrix, feature_names
