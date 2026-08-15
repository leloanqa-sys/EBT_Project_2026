import os
import glob
import zipfile
import unittest

class TestStep1Download(unittest.TestCase):

    def test_verify_downloaded_zips_and_folders(self):
        """
        Verifies Step 1:
        1. Checks that all 4 essential ZIP files exist in data/zips/ and are valid archives.
        2. Checks that data/raw/ contains extracted subfolders: clip-features-32, map-keyframes, media-info, objects.
        """
        zips_dir = "data/zips"
        raw_dir = "data/raw"

        self.assertTrue(os.path.exists(zips_dir), "❌ zips directory missing!")
        self.assertTrue(os.path.exists(raw_dir), "❌ raw directory missing!")

        zip_files = glob.glob(os.path.join(zips_dir, "*.zip"))
        print(f"\n🔍 Found {len(zip_files)} zip files in {zips_dir}")
        self.assertGreaterEqual(len(zip_files), 4, "❌ Missing Tier 1 ZIP files!")

        for z_path in zip_files:
            fname = os.path.basename(z_path)
            self.assertTrue(zipfile.is_zipfile(z_path), f"❌ File {fname} is not a valid zip archive!")
            with zipfile.ZipFile(z_path, 'r') as z:
                self.assertIsNone(z.testzip(), f"❌ Zip file {fname} is corrupted!")
            print(f"  ✓ ZIP intact: {fname}")

        expected_subfolders = ["clip-features-32", "map-keyframes", "media-info", "objects"]
        for subf in expected_subfolders:
            subf_path = os.path.join(raw_dir, subf)
            self.assertTrue(os.path.exists(subf_path), f"❌ Extracted subfolder missing: {subf_path}")
            contents = os.listdir(subf_path)
            self.assertGreater(len(contents), 0, f"❌ Extracted subfolder is empty: {subf_path}")
            print(f"  ✓ Extracted folder ready: {subf} ({len(contents)} items)")

if __name__ == "__main__":
    unittest.main()
