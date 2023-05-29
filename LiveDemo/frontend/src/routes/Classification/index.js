import { useState } from 'react'
import styles from './styles.module.css'
import ImageInputBox from '../../components/ImageInputBox'
import ModelSelection from '../../components/ModelSelection'
import ResultBox from '../../components/ResultBox'

const Home = () => {

    const [model, setModel] = useState(null)
    const [image, setImage] = useState(null)

    return (
        <div className={styles.wrapper}>
            <div className={styles.content}>
                <ModelSelection setModel={setModel} />
                <div className={styles.classificationWrapper}>
                    <ImageInputBox setImage={setImage} />
                    <ResultBox />
                </div>
            </div>
        </div>
    )
}

export default Home