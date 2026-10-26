SETUP & INTEGRATION GUIDE FOR OPTICAL-SAR MODULE

1. Place all files (model.py, preprocessing.py, validator.py, optical_sar_tool.py, and fusion_weights.pt) inside a folder named "Optical-SAR" in the root directory.

2. Install dependencies by running:
   pip install -r requirements.txt

3. How to import and use the tool inside LangGraph (tools.py):
   ----------------------------------------------------------------
   import sys
   from langchain_core.tools import tool

   # Point to the Optical-SAR directory path
   sys.path.append(r"C:\path\to\Optical-SAR")
   from optical_sar_tool import optical_sar_tool as execute_fusion_model

   @tool
   def optical_sar_fusion_tool(optical_path: str, sar_path: str, query: str) -> dict:
       """Use this tool when analyzing a co-registered Optical and SAR image pair."""
       return execute_fusion_model(optical_path, sar_path, query)
   ----------------------------------------------------------------

4. The tool automatically validates input files, resamples mismatched coordinates, performs Dual-Encoder U-Net inference using 'fusion_weights.pt', and outputs a JSON dict containing text answers, percentage metrics, a web PNG mask, and a GIS-compliant GeoTIFF (`fused_mask_georeferenced.tif`).