/* Unit Test for the FrontendSummary.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import FrontendSummary from '../Components/FrontendSummary';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() });

describe('FrontendSummary.js Test', () => {

    /* Creates a sample FrontendSummary.js component to check if it renders the correct numbet of h3, p, and button 
       tags.
    */
    it('Frontend Summary has 1 <h3> tag, 5 <p> tags, and 1 <button>.', () => {
        const minimal_verifier_summary_JSON = {
            "num_total_image_pairs": 66,
            "num_valid_image_pairs": 66,
            "rotation_success_count": 66,
            "translation_success_count": 66,
            "pose_success_count": 66,
            "num_all_inlier_correspondences_wrt_gt_model": 43,
        };
    
        const wrapper = shallow(<FrontendSummary frontend_summary={minimal_verifier_summary_JSON}/>);
        const div = wrapper.find('div[className="fs_container"]');
        const tagList = div.props().children.props.children;

        var num_h3 = 0;
        var num_p = 0;
        var num_button = 0;

        for (var i = 0; i < tagList.length; i++) {
            if (tagList[i].type == 'h3') {
                num_h3++;
            } else if (tagList[i].type == 'p') {
                num_p++;
            } else if (tagList[i].type == 'button') {
                num_button++;
            }
        }

        expect(num_h3).toEqual(1);
        expect(num_p).toEqual(5);
        expect(num_button).toEqual(1);
    })
})