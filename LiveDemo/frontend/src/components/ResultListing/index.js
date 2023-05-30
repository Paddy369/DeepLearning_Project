import { Card, CardMedia, CardContent, CircularProgress, Typography } from '@mui/material'
import styles from './styles.module.css'

const ResultListing = ({ className, percentage, isMax, state }) => {
    return (
        <Card sx={{ display: 'flex', width: 250 }} className={`${styles.card} ${isMax && styles.max}`}>
            <CardMedia
                component="img"
                sx={{ width: 90 }}
                image={`/card_thumbnails/${className}.jpg`}
                alt="Cat"
            />
            <CardContent sx={{ flex: '1 0 auto' }} className={styles.contentWrapper}>
                <Typography variant="h6" size="0.5rem" >{className}</Typography>
                <div className={styles.percentageWrapper}><Typography variant="p" className={styles.percentageLabel}>Prediction:</Typography> {
                    state == "loaded"  ? <Typography variant="p" sx={{ fontWeight: 1000 }}>{percentage}</Typography> :
                    state == "loading" ? <CircularProgress color="grey" size="1rem" /> :
                                         <Typography variant="p">-</Typography>
                } </div>
            </CardContent>
        </Card>
    )
}

export default ResultListing