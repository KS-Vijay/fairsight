from hdbcli import dbapi

# Replace with your HANA Cloud credentials
HANA_HOST = "d4749caf-d293-4be5-8cde-fdd920efefac.hana.trial-us10.hanacloud.ondemand.com"
HANA_USER = "DBADMIN"
HANA_PASSWORD = "FairSight000!"  # Set during instance creation

def test_connection():
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=443,
            user=HANA_USER,
            password=HANA_PASSWORD,
            encrypt=True  # Required for security
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 'Hello from HANA Cloud!' FROM DUMMY")
        result = cursor.fetchone()[0]
        print("✅ Connection successful! Response:", result)
        return True
    except Exception as e:
        print("❌ Connection failed:", str(e))
        return False

if __name__ == "__main__":
    test_connection()