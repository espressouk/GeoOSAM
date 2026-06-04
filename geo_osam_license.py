"""
GeoOSAM SAM3 License Management Module

This module handles license validation for SAM3 Pro features.
- Free tier: SAM3 text/similar mode on visible extent (AOI)
- Pro tier: SAM3 text/similar mode on entire raster (requires license)

License keys are validated via Cloudflare Worker with device limits.
Offline mode: Cached for 30 days after online validation.
"""

import hashlib
import hmac
import json
import urllib.request
import platform
import uuid
from datetime import datetime, timedelta
from qgis.PyQt.QtCore import QSettings


class LicenseManager:
    """Manages SAM3 license validation for GeoOSAM"""

    # Cloudflare Worker URL for license validation
    _WORKER_URL = "https://geoosam-licenses.d9s7mysbb4.workers.dev"

    # QSettings keys
    _SETTINGS_PREFIX = "GeoOSAM"
    _KEY_LICENSE_KEY = f"{_SETTINGS_PREFIX}/license_key"
    _KEY_LICENSE_EMAIL = f"{_SETTINGS_PREFIX}/license_email"
    _KEY_LICENSE_VALIDATED = f"{_SETTINGS_PREFIX}/license_validated"
    _KEY_LICENSE_TYPE = f"{_SETTINGS_PREFIX}/license_type"
    _KEY_LICENSE_CACHE_DATE = f"{_SETTINGS_PREFIX}/license_cache_date"
    _KEY_DEVICE_ID = f"{_SETTINGS_PREFIX}/device_id"

    # In-memory session cache — avoids a network round-trip on every Pro feature check.
    # None = not yet checked this session; 'pro'/'free' = result of last check.
    _session_license_type = None

    # Cache expiry (30 days)
    _CACHE_EXPIRY_DAYS = 30

    # HMAC secret for cache tamper-detection.
    # Stops QSettings manipulation — forging a valid signature without this key is hard.
    _CACHE_SECRET = b"G30OSAM$c4ch3$1nt3gr1ty$2025$v1"
    _KEY_LICENSE_CACHE_SIG = f"{_SETTINGS_PREFIX}/license_cache_sig"

    @staticmethod
    def _get_device_id():
        """
        Get unique device identifier

        Returns:
            str: Unique device ID
        """
        settings = QSettings()
        device_id = settings.value(LicenseManager._KEY_DEVICE_ID, "")

        if not device_id:
            try:
                hostname = platform.node()
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                                for elements in range(0, 2 * 6, 2)][::-1])
                system = platform.system()
                device_str = f"{hostname}:{mac}:{system}"
                device_id = hashlib.sha256(
                    device_str.encode()).hexdigest()[:16]
                settings.setValue(LicenseManager._KEY_DEVICE_ID, device_id)
            except Exception:
                device_id = str(uuid.uuid4())[:16]
                settings.setValue(LicenseManager._KEY_DEVICE_ID, device_id)

        return device_id

    @staticmethod
    def validate_license(license_key, email):
        """
        Validate license key for given email (online + offline fallback)

        Args:
            license_key (str): User-entered key (GEOOSAM3-XXXXX-XXXXX-XXXXX-XXXXX)
            email (str): User's email address

        Returns:
            bool: True if valid, False otherwise
        """
        if not license_key or not email:
            return False

        try:
            # Try online validation first
            result = LicenseManager._validate_online(license_key, email)
            if result['valid']:
                LicenseManager._cache_validation(email, license_key)
                return True
            else:
                error_msg = result.get('error', 'Unknown error')
                if 'already activated' in error_msg.lower():
                    return False

        except Exception:
            pass

        if LicenseManager._validate_from_cache(email, license_key):
            return True

        return False

    @staticmethod
    def _validate_online(license_key, email):
        """
        Validate license against Cloudflare Worker

        Args:
            license_key (str): User's license key
            email (str): User's email

        Returns:
            dict: {'valid': bool, 'message': str, 'error': str}
        """
        if LicenseManager._WORKER_URL == "REPLACE_WITH_YOUR_WORKER_URL":
            return {'valid': False, 'error': 'Worker not configured'}

        try:
            # Hash the email for privacy
            email_hash = hashlib.sha256(
                email.lower().strip().encode()).hexdigest()

            # Get device ID
            device_id = LicenseManager._get_device_id()

            # Prepare request data
            data = {
                'email_hash': email_hash,
                'license_key': license_key.upper().strip().replace(" ", ""),
                'device_id': device_id,
                'action': 'validate'
            }

            # Send POST request to worker
            req = urllib.request.Request(
                LicenseManager._WORKER_URL,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'GeoOSAM-QGIS-Plugin/1.3'
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=10) as response:  # nosec B310 - URL is hardcoded https:// license server
                result = json.loads(response.read().decode('utf-8'))
                return result

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                error_data = json.loads(error_body)
                return error_data
            except Exception:
                return {'valid': False, 'error': f'HTTP {e.code}: {error_body}'}
        except urllib.error.URLError as e:
            return {'valid': False, 'error': f'Network error: {e.reason}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}

    @staticmethod
    def _compute_cache_sig(email, license_key, device_id, date_str):
        """HMAC signature over the cache entry — detects QSettings tampering."""
        msg = f"{email.lower().strip()}:{license_key.upper().strip()}:{device_id}:{date_str}".encode()
        return hmac.new(LicenseManager._CACHE_SECRET, msg, hashlib.sha256).hexdigest()

    @staticmethod
    def _cache_validation(email, license_key):
        """Cache a successful validation for offline use."""
        try:
            settings = QSettings()
            date_str = datetime.now().isoformat()
            device_id = LicenseManager._get_device_id()
            sig = LicenseManager._compute_cache_sig(email, license_key, device_id, date_str)
            settings.setValue(LicenseManager._KEY_LICENSE_EMAIL, email)
            settings.setValue(LicenseManager._KEY_LICENSE_KEY, license_key)
            settings.setValue(LicenseManager._KEY_LICENSE_CACHE_DATE, date_str)
            settings.setValue(LicenseManager._KEY_LICENSE_CACHE_SIG, sig)
        except Exception:
            pass

    @staticmethod
    def _validate_from_cache(email, license_key):
        """
        Validate using cached data (offline mode)

        Args:
            email (str): User's email
            license_key (str): User's license key

        Returns:
            bool: True if cached validation is still valid
        """
        try:
            settings = QSettings()

            # Check if we have cached data
            cached_email = settings.value(
                LicenseManager._KEY_LICENSE_EMAIL, "")
            cached_key = settings.value(LicenseManager._KEY_LICENSE_KEY, "")
            cache_date_str = settings.value(
                LicenseManager._KEY_LICENSE_CACHE_DATE, "")

            if not cached_email or not cached_key or not cache_date_str:
                return False

            # Check if cache is expired
            try:
                cache_date = datetime.fromisoformat(cache_date_str)
                expiry_date = cache_date + \
                    timedelta(days=LicenseManager._CACHE_EXPIRY_DAYS)

                if datetime.now() > expiry_date:
                    return False
            except ValueError:
                return False

            # Verify HMAC signature — rejects any entry whose QSettings were tampered with
            stored_sig = settings.value(LicenseManager._KEY_LICENSE_CACHE_SIG, "")
            device_id = LicenseManager._get_device_id()
            expected_sig = LicenseManager._compute_cache_sig(
                cached_email, cached_key, device_id, cache_date_str)
            if not stored_sig or not hmac.compare_digest(stored_sig, expected_sig):
                return False

            # Credentials must match what was cached
            if (email.lower().strip() == cached_email.lower().strip() and  # noqa: W504
                    license_key.upper().strip().replace(" ", "") == cached_key.upper().strip().replace(" ", "")):
                return True

            return False

        except Exception:
            return False

    @staticmethod
    def get_license_type():
        """
        Get current license type.

        Returns the in-memory session result on repeated calls within the same
        QGIS session, avoiding a network round-trip on every Pro feature check.
        The session cache is invalidated when save_license() or clear_license()
        is called (i.e. when the user changes their licence state).

        Returns:
            str: 'pro' if licensed, 'free' otherwise
        """
        if LicenseManager._session_license_type is not None:
            return LicenseManager._session_license_type

        settings = QSettings()
        is_validated = settings.value(
            LicenseManager._KEY_LICENSE_VALIDATED, False, type=bool)

        result = 'free'
        if is_validated:
            stored_key = settings.value(LicenseManager._KEY_LICENSE_KEY, "")
            stored_email = settings.value(LicenseManager._KEY_LICENSE_EMAIL, "")
            if stored_key and stored_email:
                if LicenseManager.validate_license(stored_key, stored_email):
                    result = 'pro'

        LicenseManager._session_license_type = result
        return result

    @staticmethod
    def save_license(license_key, email):
        """
        Save validated license to QSettings

        Args:
            license_key (str): The license key
            email (str): User's email address

        Returns:
            bool: True if saved successfully
        """
        try:
            settings = QSettings()
            date_str = datetime.now().isoformat()
            device_id = LicenseManager._get_device_id()
            sig = LicenseManager._compute_cache_sig(email, license_key, device_id, date_str)
            settings.setValue(LicenseManager._KEY_LICENSE_KEY, license_key)
            settings.setValue(LicenseManager._KEY_LICENSE_EMAIL, email)
            settings.setValue(LicenseManager._KEY_LICENSE_VALIDATED, True)
            settings.setValue(LicenseManager._KEY_LICENSE_TYPE, 'pro')
            settings.setValue(LicenseManager._KEY_LICENSE_CACHE_DATE, date_str)
            settings.setValue(LicenseManager._KEY_LICENSE_CACHE_SIG, sig)
            LicenseManager._session_license_type = 'pro'
            return True
        except Exception:
            return False

    @staticmethod
    def load_license():
        """
        Load license from QSettings

        Returns:
            dict: {'key': str, 'email': str, 'validated': bool, 'type': str}
                  or None if no license stored
        """
        try:
            settings = QSettings()

            key = settings.value(LicenseManager._KEY_LICENSE_KEY, "")
            email = settings.value(LicenseManager._KEY_LICENSE_EMAIL, "")
            validated = settings.value(
                LicenseManager._KEY_LICENSE_VALIDATED, False, type=bool)
            license_type = settings.value(
                LicenseManager._KEY_LICENSE_TYPE, 'free')

            if key and email and validated:
                return {
                    'key': key,
                    'email': email,
                    'validated': validated,
                    'type': license_type
                }

            return None

        except Exception:
            return None

    @staticmethod
    def clear_license():
        """Remove license from QSettings"""
        try:
            settings = QSettings()
            settings.remove(LicenseManager._KEY_LICENSE_KEY)
            settings.remove(LicenseManager._KEY_LICENSE_EMAIL)
            settings.remove(LicenseManager._KEY_LICENSE_VALIDATED)
            settings.remove(LicenseManager._KEY_LICENSE_TYPE)
            settings.remove(LicenseManager._KEY_LICENSE_CACHE_DATE)
            settings.remove(LicenseManager._KEY_LICENSE_CACHE_SIG)
            # Note: Don't remove device_id - it should persist
            LicenseManager._session_license_type = None
            return True
        except Exception:
            return False

    @staticmethod
    def has_raster_access():
        """
        Check if user has access to entire raster processing

        Returns:
            bool: True if user has pro license, False otherwise
        """
        return LicenseManager.get_license_type() == 'pro'

    @staticmethod
    def get_license_info():
        """
        Get human-readable license information

        Returns:
            dict: {'type': str, 'email': str, 'status': str}
        """
        license_type = LicenseManager.get_license_type()

        if license_type == 'pro':
            license_data = LicenseManager.load_license()
            if license_data:
                return {
                    'type': 'pro',
                    'email': license_data.get('email', 'Unknown'),
                    'status': 'SAM3 Pro License Active'
                }

        return {
            'type': 'free',
            'email': None,
            'status': 'SAM3 Free Tier (Extent Only)'
        }
