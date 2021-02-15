var ViewFrustum = require('./view_frustum.js');
var SE3 = require('./se3.js');
var nj = require('numjs');
var Quaternion = require('quaternion');


sampleFrustum = new ViewFrustum(1532.24, 1920, 1080)
var q = new Quaternion(0.997524, -0.0659275, 0.011254, 0.0217253) //w,x,y,z
var rotation_matrix = q.toMatrix(true);
var rotation = nj.array(rotation_matrix);
var translation = nj.array([0.13125, -0.708675, -2.39795]);
var wTc = new SE3(rotation, translation);
var verts_worldfr = sampleFrustum.get_mesh_vertices_worldframe(wTc)
console.log(verts_worldfr[5])
