// SELECT ONE OF THE REGIONS BELOW
var region = 'CW';
var region = 'BC';

var out_scale = 100;

if (region === 'CW') {
  var assetPath = 'projects/ee-philipparndt/assets/CW-bbox-polygon';
  var dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic");
  var description = 'hillshade_export_CW_100m';
  var crs_out = 'EPSG:3413';
  var assetToClip = 'projects/ee-philipparndt/assets/GRE_basins_merged_GRIMP';
  var exaggeration = 5;
} else if (region === 'BC') {
  var assetPath = 'projects/ee-philipparndt/assets/BC-bbox-polygon';
  var dem = ee.Image("UMN/PGC/REMA/V1_1/8m");
  var description = 'hillshade_export_BC_100m';
  var crs_out = 'EPSG:3031';
  var exaggeration = 5;
  var assetToClip = 'projects/ee-philipparndt/assets/ANT_basins_merged';
}

var bufferPoly = function(feature) {
  return feature.buffer(30000);   
};

var roi = ee.FeatureCollection(assetPath).map(bufferPoly);
var toclip = ee.FeatureCollection(assetToClip);
var dem = dem.clip(roi).clip(toclip);

print(assetPath)
//print(dem)

var exaggeration = 5;
var hillshade = ee.Terrain.hillshade(dem.select('elevation').multiply(exaggeration),270,45);
print(hillshade);
var hillshade_8bit = hillshade.toUint8();

// Show the ROI and center map on it
Map.addLayer(hillshade_8bit, {min: 0, max: 255}, 'hillshade');
var styling = {color: 'red', fillColor: '00000000'};
Map.addLayer(roi.style(styling),null,'Region Of Interest');
Map.centerObject(roi, 5);

// Export the hillshade to Google Drive
Export.image.toDrive({
  image: hillshade_8bit,
  description: description,
  folder: 'EarthEngineExports',
  fileNamePrefix: description,
  region: roi.geometry().bounds(),
  scale: out_scale,
  crs: crs_out,
  maxPixels: 1e11  // Adjust as needed based on your region size
});