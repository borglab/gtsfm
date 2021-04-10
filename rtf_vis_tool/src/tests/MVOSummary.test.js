/* Unit Test for the MVOSummary.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import MVOSummary from '../Components/MVOSummary';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() });

describe('MVOSummary.js Test', () => {

    /* Creates a sample MVOSummary.js component to check if it renders the correct number of p tags.
    */
    it('MVOSummary has 12 <p> tags.', () => {
        const mvo_summary = {
            "rotation_averaging_angle_deg": {
                "median_error": 0.09,
                "min_error": 0.01,
                "max_error": 0.18
            },
            "translation_averaging_distance": {
                "median_error": 0.04,
                "min_error": 0.01,
                "max_error": 0.09
            },
            "translation_to_direction_angle_deg": {
                "median_error": 0.2,
                "min_error": 0.04,
                "max_error": 1.2
            }
        };
    
        const wrapper = shallow(<MVOSummary json={mvo_summary}/>);
        const p_tags = wrapper.find('p');
        
        expect(p_tags.length).toEqual(12);
    })
})