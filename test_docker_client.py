"""
Test script to debug Docker container issue with full traceback
"""
import traceback
from gradio_client import Client

try:
    print("Connecting to Gradio client...")
    client = Client('http://localhost:8080/')
    print("✓ Connected successfully")
    
    print("\nSending prediction request...")
    result = client.predict(
        api_key='LiFoVoxCOSfliJst1aCrnhuwsgDVSJRp',
        latent_dim=5,
        epochs=50,
        num_samples=10,  # Start with fewer materials for faster testing
        available_equipment=[
            'Muffle Furnace', 'Ball Mill', 'Glove Box', 'Arc Melter',
            'Induction Furnace', 'Resistance Furnace', 'Vacuum Arc Melter',
            'Plasma Arc Furnace', 'Vacuum Chamber', 'CVD Reactor',
            'Gas Handling System', 'Controlled Atmosphere'
        ],
        api_name='/handle_generate_materials'
    )
    
    print("\n✓ Prediction completed")
    print(f"\nResult summary:")
    print(f"  Summary text length: {len(result[0]) if result[0] else 0}")
    print(f"  Materials table rows: {len(result[2]['data']) if result[2] and 'data' in result[2] else 0}")
    print(f"  Priority table rows: {len(result[3]['data']) if result[3] and 'data' in result[3] else 0}")
    
    if result[2] and 'data' in result[2] and len(result[2]['data']) > 0:
        print("\n✅ SUCCESS: Materials were generated!")
        print(f"First material: {result[2]['data'][0][:3]}")  # Show first 3 columns
    else:
        print("\n❌ FAIL: No materials in result")
        print(f"Full result[0] (error message): {result[0]}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()