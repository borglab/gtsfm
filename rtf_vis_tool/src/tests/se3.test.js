/* Unit Test for the SE3 Class.

Author: Adi Singh
*/

// Third-Party Package Imports.
var nj = require('numjs');

// Local Imports.
var SE3 = require('../Components/frustum_classes/se3');

describe('SE3 Class Unit Tests', () => {

    /* Creates an SE3 constructor and verifies it. */
    it('Test SE3 Constructor.', () => {
        var sample_rot = nj.array([[1,1,1],
                                   [2,2,2],
                                   [3,3,3]]);
        var sample_translation = nj.array([1,2,3]);

        const computed_se3_obj = new SE3(sample_rot, sample_translation);
        const expected_se3_obj = nj.array([[1,1,1,1],
                                         [2,2,2,2],
                                         [3,3,3,3],
                                         [0,0,0,1]]);
        console.log(expected_se3_obj)
        expect(expected_se3_obj).toEqual(computed_se3_obj.transform_matrix);
    });
})