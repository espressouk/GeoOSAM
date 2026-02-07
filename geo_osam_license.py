"""
GeoOSAM SAM3 License Management Module

This module handles license validation for SAM3 Pro features.
- Free tier: SAM3 text/similar mode on visible extent (AOI)
- Pro tier: SAM3 text/similar mode on entire raster (requires license)

License keys are validated via Cloudflare Worker with device limits.
Offline mode: Cached for 30 days after online validation.
"""

import hashlib
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

    # Cache expiry (30 days)
    _CACHE_EXPIRY_DAYS = 30

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
            # Generate device ID from machine characteristics
            try:
                # Combine hostname + MAC address + platform
                hostname = platform.node()
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                                for elements in range(0, 2*6, 2)][::-1])
                system = platform.system()

                # Hash to create stable device ID
                device_str = f"{hostname}:{mac}:{system}"
                device_id = hashlib.sha256(
                    device_str.encode()).hexdigest()[:16]

                # Save for future use
                settings.setValue(LicenseManager._KEY_DEVICE_ID, device_id)
                print(f"Generated device ID: {device_id}")
            except Exception as e:
                print(f"⚠️  Error generating device ID, using fallback: {e}")
                # Fallback to random UUID (less stable but works)
                device_id = str(uuid.uuid4())[:16]
                settings.setValue(LicenseManager._KEY_DEVICE_ID, device_id)

        return device_id

    @staticmethod
    def validate_license(license_key, email):
        """
        Validate license key for given email (online + offline fallback)

        Args:
            license_key (str): User-entered key (GEOSAM3-XXXXX-XXXXX-XXXXX-XXXXX)
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
                print(
                    f"✅ License validated online: {result.get('message', '')}")
                # Cache the validation
                LicenseManager._cache_validation(email, license_key)
                return True
            else:
                # Show error from server
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ Online validation failed: {error_msg}")

                # If device limit reached, don't try cache
                if 'already activated' in error_msg.lower():
                    return False

        except Exception as e:
            print(f"⚠️  Online validation failed, trying offline cache: {e}")

        # Fallback to cached validation (offline mode)
        if LicenseManager._validate_from_cache(email, license_key):
            print("✅ License validated from cache (offline mode)")
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
            print("⚠️  Worker URL not configured - using offline mode only")
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

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                error_data = json.loads(error_body)
                return error_data
            except:
                return {'valid': False, 'error': f'HTTP {e.code}: {error_body}'}
        except urllib.error.URLError as e:
            return {'valid': False, 'error': f'Network error: {e.reason}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}

    @staticmethod
    def _cache_validation(email, license_key):
        """
        Cache a successful validation for offline use

        Args:
            email (str): User's email
            license_key (str): Valid license key
        """
        try:
            settings = QSettings()
            settings.setValue(
                LicenseManager._KEY_LICENSE_CACHE_DATE, datetime.now().isoformat())
            print(f"✅ License cached for offline use (30 days)")
        except Exception as e:
            print(f"⚠️  Failed to cache license: {e}")

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
                    print(
                        "⚠️  Cached license expired (>30 days) - please connect to internet to revalidate")
                    return False
            except ValueError:
                return False

            # Compare with cached credentials
            if (email.lower().strip() == cached_email.lower().strip() and
                    license_key.upper().strip().replace(" ", "") == cached_key.upper().strip().replace(" ", "")):
                days_remaining = (expiry_date - datetime.now()).days
                print(
                    f"ℹ️  Using cached license ({days_remaining} days remaining)")
                return True

            return False

        except Exception as e:
            print(f"⚠️  Cache validation error: {e}")
            return False

    @staticmethod
    def get_license_type():
        """
        Get current license type

        Returns:
            str: 'pro' if licensed, 'free' otherwise
        """
        settings = QSettings()

        # Check if license is validated
        is_validated = settings.value(
            LicenseManager._KEY_LICENSE_VALIDATED, False, type=bool)

        if is_validated:
            # Double-check by re-validating stored credentials (uses cache if offline)
            stored_key = settings.value(LicenseManager._KEY_LICENSE_KEY, "")
            stored_email = settings.value(
                LicenseManager._KEY_LICENSE_EMAIL, "")

            if stored_key and stored_email:
                if LicenseManager.validate_license(stored_key, stored_email):
                    return 'pro'

        return 'free'

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
            settings.setValue(LicenseManager._KEY_LICENSE_KEY, license_key)
            settings.setValue(LicenseManager._KEY_LICENSE_EMAIL, email)
            settings.setValue(LicenseManager._KEY_LICENSE_VALIDATED, True)
            settings.setValue(LicenseManager._KEY_LICENSE_TYPE, 'pro')
            settings.setValue(
                LicenseManager._KEY_LICENSE_CACHE_DATE, datetime.now().isoformat())
            print(f"✅ License saved for {email}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to save license: {e}")
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

        except Exception as e:
            print(f"⚠️  Failed to load license: {e}")
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
            # Note: Don't remove device_id - it should persist
            print("✅ License cleared")
            return True
        except Exception as e:
            print(f"⚠️  Failed to clear license: {e}")
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
