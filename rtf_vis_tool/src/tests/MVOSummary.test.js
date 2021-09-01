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
        const averagingMetricsJSON = {
            "averaging_metrics": {
                "rotation_averaging_angle_deg": {
                    "summary": {
                        "median": 0.09,
                        "min": 0.01,
                        "max": 0.18
                    }
                },
                "translation_averaging_distance": {
                    "summary": {
                        "median": 0.04,
                        "min": 0.01,
                        "max": 0.09
                    }
                },
                "translation_angle_deg": {
                    "summary": {
                        "median": 0.2,
                        "min": 0.04,
                        "max": 1.2
                    }
                }
            }
        }
    
        const wrapper = shallow(<MVOSummary json={averagingMetricsJSON}/>);
        const p_tags = wrapper.find('p');
        
        expect(p_tags.length).toEqual(12);
    })
})