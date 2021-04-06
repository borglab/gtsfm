/* Dependencies */
import React from 'react'
import Enzyme, { shallow } from 'enzyme'
import Adapter from 'enzyme-adapter-react-16'

/* Component to Test*/
import SfMDataDivNode from './Components/SfMDataDivNode';

// Configure enzyme for react 16
Enzyme.configure({ adapter: new Adapter() })

describe('SfMData DivNode Test', () => {
    it('should render `Node Testing` inside the DivNode component', () => {
        const sample_json = {'num': 50};
        const sampleFunction = () => console.log('sample function');
        const wrapper = shallow(<SfMDataDivNode 
                                    json={sample_json} 
                                    toggleDA_PC={sampleFunction} 
                                    textColor={'black'} 
                                    backgroundColor={'gray'} 
                                    topOffset={'10%'} 
                                    leftOffset={'10%'} 
                                    text={'GtsfmData'}/>);
        const item = wrapper.find('div[class="standard_div_node_style"]');
        expect(item).toHaveLength(1);
        expect(item.first().text()).toEqual('GtsfmData');
    })
})